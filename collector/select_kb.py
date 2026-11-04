"""① 수집 계층 — 지식베이스 선별 (store/docs.db의 documents.in_kb 표시).

수집한 문서는 모두 docs.db에 남기고, 그중 인덱서가 임베딩할 문서에만 in_kb=1과 선별 사유(kb_reason)를 붙인다.
6일 안에 임베딩·평가·개선을 반복할 수 있는 규모(문서 약 15,000건 / 청크 약 3만 개)로 맞추면서,
직접 수집한 1차 자료(공시·특허·보고서)의 비중이 유의미하도록 구성한다. (README §8)

  뉴스(제공)  : 근사중복 클러스터의 대표본(가장 이른 기사)만 → 점수(하위 주제·전망 서술·키워드 밀도) 순으로
                분기별 층화 추출해 NEWS_TARGET건
  DART 공시   : 수시공시 전부 + 사업보고서는 최근 ANNUAL_YEARS 사업연도만
  특허        : 해외 전부 + 국내는 DART 시드 기업 출원 전부 + 주제별 할당량(PATENT_QUOTA, 등록·최근 우선)
  산업보고서  : BioIN 분석 보고서(BioINdustry·BioINpro·BioINglobal) 전부 + 산업 동향 브리프는
                점수(하위 주제·전망 서술) 순 REPORT_BRIEF_TARGET건
  그 외 출처  : 전부

크롤러가 문서를 다시 적재하면 in_kb가 초기화되므로, 수집 후에는 항상 이 단계를 다시 실행한다.

실행:
    python -m collector.select_kb
    python -m collector.select_kb --news-target 6000      # 일정이 밀리면 뉴스부터 축소
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from collector import storage

log = logging.getLogger("collector.select_kb")

NEWS_TARGET = 8000
NEAR_DUP_THRESHOLD = 0.8          # MinHash 시그니처 일치율 (문자 5-gram, 본문 앞 1,500자)
ANNUAL_YEARS = ("2024.12", "2025.12")
PATENT_QUOTA = {                  # 국내 특허 주제별 상한 (None = 전부). 부족한 하위 주제는 전부 유지
    "ADC": None,
    "AI 신약개발": None,
    "의료 AI 규제": 600,
    "비대면 진료": 350,
}
REPORT_BRIEF_TARGET = 500         # BioIN '산업 동향'(유관기관 브리프) 상한
CHUNK_CHARS = 700                 # 규모 추정용 청크 길이


# --------------------------------------------------------------------------- #
# 근사중복
# --------------------------------------------------------------------------- #
def near_dup_clusters(texts: list[str], threshold: float = NEAR_DUP_THRESHOLD, n_perm: int = 64, bands: int = 16) -> list[int]:
    """MinHash + LSH로 근사중복 클러스터 번호(대표 인덱스)를 돌려준다."""
    rows = n_perm // bands
    rng = random.Random(0)
    prime = (1 << 61) - 1
    coeffs = [(rng.randrange(1, prime), rng.randrange(0, prime)) for _ in range(n_perm)]

    def sig(t: str) -> list[int]:
        t = re.sub(r"[^가-힣A-Za-z0-9]", "", t)[:1500]
        shingles = {int(hashlib.md5(t[i:i + 5].encode()).hexdigest()[:12], 16) for i in range(0, max(1, len(t) - 4), 2)} or {0}
        return [min((a * x + b) % prime for x in shingles) for a, b in coeffs]

    sigs = [sig(t) for t in texts]
    parent = list(range(len(sigs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in range(bands):
        buckets: dict[tuple, list[int]] = defaultdict(list)
        for i, s in enumerate(sigs):
            buckets[tuple(s[b * rows:(b + 1) * rows])].append(i)
        for idx in buckets.values():
            for j in idx[1:]:
                i = idx[0]
                if sum(x == y for x, y in zip(sigs[i], sigs[j])) / n_perm >= threshold:
                    parent[find(j)] = find(i)
    return [find(i) for i in range(len(sigs))]


def _tiebreak(doc_id: str) -> int:
    return int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# 출처별 선별
# --------------------------------------------------------------------------- #
def select_news(df: pd.DataFrame, target: int) -> pd.Series:
    """반환: doc_id → kb_reason (선별된 것만)."""
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["cluster"] = near_dup_clusters(df["body"].tolist())
    reps = df.sort_values("published_at").drop_duplicates("cluster")
    log.info("뉴스 %d건 → 근사중복 대표본 %d건", len(df), len(reps))

    reps = reps.assign(
        score=2 * (reps["topics"].fillna("") != "")
        + 1 * reps["is_outlook"].fillna(0)
        + 1 * (reps["keyword_hits"].fillna(0) >= 2)
        + 0.5 * reps["char_len"].between(600, 4000),
        quarter=reps["published_at"].dt.to_period("Q"),
        tb=reps["doc_id"].map(_tiebreak),
    )
    # 분기별 층화: 분기 기사량 비율대로 할당 후 점수 순
    share = reps["quarter"].value_counts(normalize=True)
    alloc = (share * target).round().astype(int)
    picked = []
    for q, k in alloc.items():
        part = reps[reps["quarter"] == q].sort_values(["score", "tb"], ascending=[False, True]).head(k)
        picked.append(part)
    out = pd.concat(picked)
    reason = out.apply(
        lambda r: "news:" + ("topic" if r["topics"] else "outlook" if r["is_outlook"] else "keyword") + f":{r['quarter']}",
        axis=1,
    )
    return pd.Series(reason.values, index=out["doc_id"])


def select_dart(df: pd.DataFrame, annual_years: tuple[str, ...]) -> pd.Series:
    meta = df["meta"].map(json.loads)
    is_annual = meta.map(lambda m: m.get("category") == "사업보고서")
    period = meta.map(lambda m: (re.search(r"\((\d{4}\.\d{2})\)", m.get("report_nm", "")) or [None, ""])[1])
    keep = ~is_annual | period.isin(annual_years)
    reason = is_annual.map({True: "dart:annual_recent", False: "dart:filing"})
    return pd.Series(reason[keep].values, index=df.loc[keep, "doc_id"])


def select_patents(df: pd.DataFrame, quota: dict[str, int | None]) -> pd.Series:
    from collector.crawl_dart import SEED_COMPANIES

    meta = df["meta"].map(json.loads)
    df = df.assign(
        country=meta.map(lambda m: m.get("country", "KR")),
        registered=meta.map(lambda m: m.get("register_status") == "등록").astype(int),
        enriched=meta.map(lambda m: "family_size" in m).astype(int),
        app_date=meta.map(lambda m: m.get("application_date") or ""),
    )
    reasons: dict[str, str] = {}

    for d in df.loc[df["country"] != "KR", "doc_id"]:
        reasons[d] = "patent:foreign"

    kr = df[df["country"] == "KR"]
    seeds = [n for ns in SEED_COMPANIES.values() for n in ns]
    seed_re = re.compile("|".join(map(re.escape, seeds)))
    for d in kr.loc[kr["publisher"].fillna("").str.contains(seed_re), "doc_id"]:
        reasons.setdefault(d, "patent:seed_applicant")

    kr = kr.sort_values(["enriched", "registered", "app_date"], ascending=False)
    for topic, cap in quota.items():
        hit = kr[kr["collect_keywords"].fillna("").str.split("; ").map(lambda ks: topic in ks)]
        hit = hit[~hit["doc_id"].isin(reasons)]
        if cap is not None:
            already = sum(1 for d, r in reasons.items() if r == f"patent:{topic}")
            hit = hit.head(max(0, cap - already))
        for d in hit["doc_id"]:
            reasons[d] = f"patent:{topic}"
    # 할당 주제 외 검색식(CDMO·바이오시밀러 등)으로 들어온 소량 특허는 전부
    rest = kr[~kr["doc_id"].isin(reasons) & ~kr["collect_keywords"].fillna("").str.contains("|".join(quota))]
    for d in rest["doc_id"]:
        reasons[d] = "patent:other_topic"
    return pd.Series(reasons)


def select_reports(df: pd.DataFrame, brief_target: int) -> pd.Series:
    meta = df["meta"].map(json.loads)
    board = meta.map(lambda m: m.get("board", ""))
    reasons: dict[str, str] = {}
    for d, b in zip(df["doc_id"], board):
        if b in ("watch", "report", "global"):
            reasons[d] = f"report:{b}"
    briefs = df[board == "industry"].assign(
        score=lambda x: 2 * (x["topics"].fillna("") != "") + 1 * x["is_outlook"].fillna(0),
        tb=lambda x: x["doc_id"].map(_tiebreak),
    ).sort_values(["score", "published_at", "tb"], ascending=[False, False, True])
    for d in briefs["doc_id"].head(brief_target):
        reasons[d] = "report:brief"
    return pd.Series(reasons)


# --------------------------------------------------------------------------- #
# 실행
# --------------------------------------------------------------------------- #
def run(
    db_path: Path = storage.DEFAULT_DB,
    news_target: int = NEWS_TARGET,
    dry_run: bool = False,
    brief_target: int = REPORT_BRIEF_TARGET,
) -> dict:
    with storage.connect(db_path) as con:
        docs = pd.read_sql(
            "SELECT doc_id, origin, source_type, title, body, published_at, publisher, collect_keywords, "
            "topics, is_outlook, keyword_hits, char_len, meta FROM documents",
            con,
        )

    parts = []
    by = docs.groupby("source_type")
    if "news" in by.groups:
        parts.append(select_news(by.get_group("news"), news_target))
    if "disclosure" in by.groups:
        parts.append(select_dart(by.get_group("disclosure"), ANNUAL_YEARS))
    if "patent" in by.groups:
        parts.append(select_patents(by.get_group("patent"), PATENT_QUOTA))
    if "report" in by.groups:
        parts.append(select_reports(by.get_group("report"), brief_target))
    others = docs[~docs["source_type"].isin(["news", "disclosure", "patent", "report"])]
    if len(others):
        parts.append(pd.Series("other:all", index=others["doc_id"]))
    selected = pd.concat(parts)

    kb = docs[docs["doc_id"].isin(selected.index)].assign(kb_reason=lambda d: d["doc_id"].map(selected))
    chunks = (kb["char_len"] / CHUNK_CHARS).clip(lower=1).round()
    primary = kb["source_type"] != "news"
    stats = {
        "total_docs": len(docs),
        "kb_docs": len(kb),
        "kb_by_source": kb["source_type"].value_counts().to_dict(),
        "kb_by_reason": kb["kb_reason"].str.replace(r"^(news:[a-z]+):.*$", r"\1", regex=True).value_counts().to_dict(),
        "primary_share_docs": round(primary.mean(), 3),
        "est_chunks": int(chunks.sum()),
        "est_chunks_by_source": chunks.groupby(kb["source_type"]).sum().astype(int).to_dict(),
        "primary_share_chunks": round(chunks[primary].sum() / chunks.sum(), 3),
        "kb_topics": kb["topics"].fillna("").str.split("; ").explode().loc[lambda s: s != ""].value_counts().to_dict(),
        "generated_at": storage.now_iso(),
    }

    if not dry_run:
        with storage.connect(db_path) as con:
            con.execute("UPDATE documents SET in_kb = 0, kb_reason = NULL")
            con.executemany(
                "UPDATE documents SET in_kb = 1, kb_reason = ? WHERE doc_id = ?",
                [(r, d) for d, r in selected.items()],
            )
        (Path(db_path).parent / "kb_selection_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="지식베이스 선별 → documents.in_kb")
    ap.add_argument("--db", type=Path, default=storage.DEFAULT_DB)
    ap.add_argument("--news-target", type=int, default=NEWS_TARGET)
    ap.add_argument("--brief-target", type=int, default=REPORT_BRIEF_TARGET)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.db, args.news_target, args.dry_run, args.brief_target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

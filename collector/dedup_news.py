"""① 수집 계층 — 뉴스 중복 처리 (3단계).

제공 뉴스는 키워드 기준으로 수집돼 같은 보도자료를 매체마다 다시 쓴 기사가 많다. 그대로 두면 검색 상위가
같은 내용으로 채워지지만, 같은 사건이라도 관점이나 수치가 다른 보도까지 지우면 상충 정보를 보여줄 수 없다.
그래서 '내용이 같은 것'만 줄이고 '다른 정보를 가진 것'은 남긴다.

  1. 완전 중복   URL(언론사ID+기사ID) 또는 정규화한 본문이 일치 → 1건만 유지
                 (collector/load_provided.py에서 처리, 건수는 rejected_docs에서 집계)
  2. 재배포본    본문 유사도(문자 5-gram Jaccard) ≥ REPUB_JACCARD 이고 발행 간격 ≤ REPUB_MAX_DAYS
                 또는 포함도 ≥ REPUB_CONTAINMENT·유사도 ≥ REPUB_JACCARD_WITH_CONT
                 또는 유사도 ≥ REPUB_JACCARD_ANYTIME (기간 무관 — 공지 재게재 등)
                 → 클러스터 대표 1건 유지, 나머지 제외
  3. 동일 사건   발행 간격 ≤ EVENT_MAX_DAYS 이고
                 (유사도 ≥ EVENT_JACCARD_MIN·제목 단어 겹침 ≥ EVENT_TITLE_MIN) 또는 유사도 ≥ EVENT_JACCARD_ALONE
                 → 같은 event_id로 묶는다. 짧은 기사가 긴 기사에 포함도 ≥ EVENT_CONTAINMENT로 들어 있고
                   새로운 수치가 없으면 '내용 동일'로 축약, 아니면 '관점 상이'로 유지

  공통 안전장치: 제외 후보가 대표 문서에 없는 수치(금액·비율·건수 등)를 NEW_NUMBERS_MIN개 이상 담고 있으면 제외하지 않는다.
                 수치 차이는 출처 간 상충을 보여줄 핵심 근거이기 때문이다.

결과는 documents와 같은 DB의 news_dedup 테이블과 store/dedup_stats.json에 남고,
collector/select_kb.py는 keep=1인 기사만 지식베이스 후보로 쓴다.

실행:
    python -m collector.dedup_news            # news_dedup 갱신 + 통계
    python -m collector.dedup_news --dry-run  # 통계만
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from collector import storage

log = logging.getLogger("collector.dedup_news")

# --- 임계값 (근거: README §지식베이스 구성 / docs/kb_composition.md) ------------------
SHINGLE = 5                       # 문자 5-gram (한글 2~3어절 단위)
SAMPLE_MOD = 4                    # 해시값 % 4 == 0인 shingle만 사용 (Jaccard 불편 추정, 메모리 1/4)
BODY_CAP = 3000                   # 본문 앞 3,000자 (중앙값 1,100자, 95% 3,000자)

REPUB_JACCARD = 0.50
REPUB_JACCARD_WITH_CONT = 0.40
REPUB_CONTAINMENT = 0.80
REPUB_MAX_DAYS = 3
REPUB_JACCARD_ANYTIME = 0.80

EVENT_MAX_DAYS = 1
EVENT_JACCARD_MIN = 0.15
EVENT_TITLE_MIN = 0.30
EVENT_JACCARD_ALONE = 0.30
EVENT_CONTAINMENT = 0.70
NEW_NUMBERS_MIN = 2              # 대표에 없는 수치가 이만큼 있어야 '새 정보'로 보고 유지 (1개는 단위 환산 차이가 대부분)

LSH_PERM, LSH_BANDS = 64, 16

NUM_RE = re.compile(
    r"\d[\d,.]*(?:\s*(?:조|천|백)\s*\d[\d,.]*)*\s*(?:조|억|만|천)?\s*(?:%|퍼센트|달러|유로|위안|엔|원|명|건|개|곳|배)"
    r"|\d[\d,.]*(?:\s*(?:조|천|백)\s*\d[\d,.]*)*\s*(?:조|억|만)"
)

DDL = """
CREATE TABLE IF NOT EXISTS news_dedup (
    doc_id    TEXT PRIMARY KEY,
    keep      INTEGER NOT NULL,      -- 1 = 지식베이스 후보로 유지
    tier      TEXT NOT NULL,         -- unique | republish_rep | republish_dup | republish_kept_numbers
                                     -- | event_distinct | event_collapsed
    rep_id    TEXT,                  -- 제외·축약된 경우 대신 남는 문서
    event_id  TEXT,                  -- 동일 사건 묶음 (유지된 기사끼리도 공유 → 검색 다양화·상충 비교에 사용)
    sim       REAL,                  -- 대표와의 본문 Jaccard
    updated_at TEXT
);
"""


# --------------------------------------------------------------------------- #
# 특징량
# --------------------------------------------------------------------------- #
def _norm(t: str) -> str:
    return re.sub(r"[^가-힣A-Za-z0-9]", "", t or "")


def shingle_set(text: str) -> np.ndarray:
    t = _norm(text)[:BODY_CAP]
    hs = {
        int.from_bytes(hashlib.blake2b(t[i:i + SHINGLE].encode(), digest_size=4).digest(), "little")
        for i in range(max(1, len(t) - SHINGLE + 1))
    }
    return np.array(sorted(h for h in hs if h % SAMPLE_MOD == 0), dtype=np.uint64)


def title_tokens(title: str) -> set[str]:
    return set(re.findall(r"[가-힣A-Za-z0-9]{2,}", re.sub(r"\[[^\]]*\]", "", title or "")))


def numbers(text: str) -> frozenset[str]:
    out = set()
    for m in NUM_RE.findall(text or ""):
        v = re.sub(r"[\s,]", "", m)
        v = re.sub(r"(?<=[조억만천])원$", "", v).replace("퍼센트", "%")
        out.add(v)
    return frozenset(out)


def has_new_numbers(mine: frozenset[str], ref: frozenset[str]) -> bool:
    return len(mine - ref) >= NEW_NUMBERS_MIN


def _minhash(sets: list[np.ndarray]) -> np.ndarray:
    p = np.uint64((1 << 31) - 1)
    rs = np.random.RandomState(0)
    a = rs.randint(1, int(p), LSH_PERM).astype(np.uint64)
    b = rs.randint(0, int(p), LSH_PERM).astype(np.uint64)
    out = np.zeros((len(sets), LSH_PERM), dtype=np.uint64)
    for i, s in enumerate(sets):
        if len(s):
            out[i] = ((a[:, None] * (s[None, :] % p) + b[:, None]) % p).min(1)
    return out


# --------------------------------------------------------------------------- #
# 후보 쌍 → 유사도
# --------------------------------------------------------------------------- #
def candidate_pairs(sets: list[np.ndarray], ts: np.ndarray) -> set[tuple[int, int]]:
    """LSH(기간 무관 고유사 쌍) + 발행 1일 이내 모든 쌍."""
    pairs: set[tuple[int, int]] = set()
    sig = _minhash(sets)
    rows = LSH_PERM // LSH_BANDS
    for band in range(LSH_BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        for i, row in enumerate(sig[:, band * rows:(band + 1) * rows]):
            buckets[row.tobytes()].append(i)
        for idx in buckets.values():
            if 1 < len(idx) < 200:                       # 빈 본문 등 거대 버킷 방지
                pairs.update((idx[x], idx[y]) for x in range(len(idx)) for y in range(x + 1, len(idx)))
    window = int(EVENT_MAX_DAYS * 86400)
    for i in range(len(ts)):
        j = i + 1
        while j < len(ts) and ts[j] - ts[i] <= window:
            pairs.add((i, j))
            j += 1
    return pairs


def pair_table(df: pd.DataFrame) -> pd.DataFrame:
    sets = [shingle_set(b) for b in df["body"]]
    toks = [title_tokens(t) for t in df["title"]]
    ts = df["ts"].values.astype("datetime64[s]").astype(np.int64)
    lens = np.array([len(s) for s in sets])
    rows = []
    for i, j in candidate_pairs(sets, ts):
        if not lens[i] or not lens[j]:
            continue
        inter = len(np.intersect1d(sets[i], sets[j], assume_unique=True))
        if not inter:
            continue
        jac = inter / (lens[i] + lens[j] - inter)
        tt = len(toks[i] & toks[j]) / max(1, len(toks[i] | toks[j]))
        if jac < EVENT_JACCARD_MIN and tt < EVENT_TITLE_MIN:
            continue
        rows.append((i, j, jac, inter / min(lens[i], lens[j]), tt, abs(ts[j] - ts[i]) / 86400))
    return pd.DataFrame(rows, columns=["i", "j", "jac", "cont", "tt", "days"])


# --------------------------------------------------------------------------- #
# 판정
# --------------------------------------------------------------------------- #
class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(b)] = self.find(a)


def is_republish(r) -> bool:
    return (
        (r.days <= REPUB_MAX_DAYS and (r.jac >= REPUB_JACCARD or (r.cont >= REPUB_CONTAINMENT and r.jac >= REPUB_JACCARD_WITH_CONT)))
        or r.jac >= REPUB_JACCARD_ANYTIME
    )


def is_same_event(r) -> bool:
    return r.days <= EVENT_MAX_DAYS and (
        (r.jac >= EVENT_JACCARD_MIN and r.tt >= EVENT_TITLE_MIN) or r.jac >= EVENT_JACCARD_ALONE
    )


def dedup(df: pd.DataFrame, pairs: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["published_at"])
    df = df.sort_values(["ts", "doc_id"]).reset_index(drop=True)
    if pairs is None:
        pairs = pair_table(df)
    nums = [numbers(b) for b in df["body"]]
    n = len(df)
    # 대표 선정 우선순위: 수치가 많을수록 → 본문이 길수록 → 먼저 보도된 것
    rank = df.assign(nn=[len(x) for x in nums], ln=df["body"].str.len()).sort_values(
        ["nn", "ln", "ts"], ascending=[False, False, True]).index
    prio = {i: k for k, i in enumerate(rank)}

    tier = ["unique"] * n
    rep = [None] * n
    sim = [None] * n
    jac_of = {(int(r.i), int(r.j)): r.jac for r in pairs.itertuples()}

    # 2) 재배포본
    rp = pairs[pairs.apply(is_republish, axis=1)]
    uf = _UF(n)
    for r in rp.itertuples():
        uf.union(int(r.i), int(r.j))
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)
    for members in clusters.values():
        if len(members) == 1:
            continue
        head = min(members, key=prio.get)
        tier[head] = "republish_rep"
        for m in members:
            if m == head:
                continue
            rep[m] = head
            sim[m] = jac_of.get((min(m, head), max(m, head)))
            # 대표에 없는 수치가 있으면 유지 (상충 근거 보존)
            tier[m] = "republish_kept_numbers" if has_new_numbers(nums[m], nums[head]) else "republish_dup"
    alive = [t != "republish_dup" for t in tier]

    # 3) 동일 사건
    ev = pairs[pairs.apply(is_same_event, axis=1) & ~pairs.apply(is_republish, axis=1)]
    ev = ev[[alive[int(i)] and alive[int(j)] for i, j in zip(ev["i"], ev["j"])]]
    euf = _UF(n)
    for r in ev.itertuples():
        euf.union(int(r.i), int(r.j))
    event_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if alive[i]:
            event_members[euf.find(i)].append(i)
    event_id = [None] * n
    for root, members in event_members.items():
        if len(members) > 1:
            eid = "ev:" + df.at[min(members, key=lambda x: df.at[x, "ts"]), "doc_id"]
            for m in members:
                event_id[m] = eid
                if tier[m] == "unique":
                    tier[m] = "event_distinct"
    for i in range(n):  # 재배포본 클러스터는 대표의 event_id를 따른다
        if rep[i] is not None and event_id[i] is None:
            event_id[i] = event_id[rep[i]]
    # 내용 동일 판정: 짧은 쪽이 긴 쪽에 거의 포함되고 새 수치가 없으면 축약 (긴 쪽을 남김)
    for r in ev.sort_values("cont", ascending=False).itertuples():
        i, j = int(r.i), int(r.j)
        if r.cont < EVENT_CONTAINMENT:
            continue
        short, long_ = (i, j) if len(df.at[i, "body"]) <= len(df.at[j, "body"]) else (j, i)
        if tier[short] == "event_collapsed" or tier[long_] == "event_collapsed":
            continue
        if has_new_numbers(nums[short], nums[long_]):
            continue
        tier[short], rep[short], sim[short] = "event_collapsed", long_, r.jac

    out = pd.DataFrame({
        "doc_id": df["doc_id"],
        "keep": [int(t not in ("republish_dup", "event_collapsed")) for t in tier],
        "tier": tier,
        "rep_id": [df.at[x, "doc_id"] if x is not None else None for x in rep],
        "event_id": event_id,
        "sim": [round(s, 3) if s is not None else None for s in sim],
    })
    tc = out["tier"].value_counts().to_dict()
    events = out.dropna(subset=["event_id"])
    kept_events = events[events["keep"] == 1].groupby("event_id").size()
    stats = {
        "input_docs": n,
        "candidate_pairs_scored": len(pairs),
        "republish_pairs": len(rp),
        "same_event_pairs": len(ev),
        "tiers": tc,
        "republish_clusters": int(tc.get("republish_rep", 0)),
        "removed_republish": int(tc.get("republish_dup", 0)),
        "kept_republish_with_new_numbers": int(tc.get("republish_kept_numbers", 0)),
        "removed_event_same_content": int(tc.get("event_collapsed", 0)),
        "event_groups": int(events["event_id"].nunique()),
        "event_groups_multi_view": int((kept_events > 1).sum()),
        "output_docs": int(out["keep"].sum()),
        "thresholds": {k: v for k, v in globals().items() if re.match(r"^(REPUB|EVENT|SHINGLE|SAMPLE_MOD|BODY_CAP|NEW_NUMBERS)", k)},
    }
    return out, stats


def exact_dup_stats(db_path: Path) -> dict:
    """1단계(완전 중복)는 load_provided에서 처리됐으므로 기록만 모은다."""
    with storage.connect(db_path) as con:
        rej = dict(con.execute(
            "SELECT drop_reason, COUNT(*) FROM rejected_docs WHERE origin='provided' GROUP BY 1").fetchall())
    stats_path = Path(db_path).parent / "load_provided_stats.json"
    lp = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    return {
        "raw_rows": lp.get("raw_rows"),
        "after_url_merge": lp.get("unique_docs"),
        "removed_same_url": (lp.get("raw_rows") or 0) - (lp.get("unique_docs") or 0),
        "removed_same_body": rej.get("exact_dup_body", 0),
    }


def run(db_path: Path = storage.DEFAULT_DB, dry_run: bool = False) -> dict:
    t0 = time.time()
    with storage.connect(db_path) as con:
        df = pd.read_sql(
            "SELECT doc_id, title, body, published_at, publisher FROM documents WHERE source_type = 'news'", con)
    out, stats = dedup(df)
    stats = {"exact": exact_dup_stats(db_path), **stats, "elapsed_sec": round(time.time() - t0),
             "generated_at": storage.now_iso()}
    if not dry_run:
        with storage.connect(db_path) as con:
            con.executescript(DDL)
            con.execute("DELETE FROM news_dedup")
            now = storage.now_iso()
            con.executemany(
                "INSERT INTO news_dedup (doc_id, keep, tier, rep_id, event_id, sim, updated_at) VALUES (?,?,?,?,?,?,?)",
                [(*r, now) for r in out.astype(object).where(out.notna(), None).itertuples(index=False, name=None)],
            )
        (Path(db_path).parent / "dedup_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        # 감사용: 제외·축약·유지 판정을 받은 기사와 대표 기사 제목을 나란히
        audit = out[out["tier"] != "unique"].merge(df[["doc_id", "published_at", "publisher", "title"]], on="doc_id")
        audit = audit.merge(df[["doc_id", "title"]].rename(columns={"doc_id": "rep_id", "title": "rep_title"}),
                            on="rep_id", how="left").sort_values(["event_id", "published_at"])
        audit_path = storage.PROJECT_ROOT / "data" / "compliance" / "news_dedup_decisions.csv"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="뉴스 3단계 중복 처리 → news_dedup")
    ap.add_argument("--db", type=Path, default=storage.DEFAULT_DB)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.db, args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

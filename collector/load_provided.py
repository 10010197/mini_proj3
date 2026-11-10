"""① 수집 계층 — 제공 데이터 적재 및 선별.

제공받은 바이오·헬스케어 뉴스 CSV(키워드별 파일 + 묶음 파일)를 읽어
  1) 컬럼 표준화 · URL 기반 문서 ID 부여
  2) 파일 간/파일 내 중복 병합 (수집키워드는 합집합으로 보존)
  3) 본문 정제 (포털 보일러플레이트 제거)
  4) 선별 규칙 적용 (빈 본문, 사진기사, 게시판·부고, 비뉴스 섹션, 주제 무관, 본문 완전중복)
  5) 하위 주제 태깅 (ADC·CDMO·AI 신약개발·바이오 투자·기술수출·의료 AI 규제 …)
후 store/docs.db 의 documents 테이블에 원문·메타데이터를 적재한다.

선별에서 제외된 문서는 rejected_docs 테이블에 사유와 함께 남겨 EDA·리포트에서 추적할 수 있게 한다.
근사 중복(유사 본문) 처리와 청킹·임베딩은 ② 인덱서(indexer/dedup.py 등)의 책임이다.

실행 예:
    python -m collector.load_provided --src "C:/Users/student/Downloads"
    python -m collector.load_provided --src data/provided --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from collector import storage
from collector.storage import EVIDENCE_ROLE

log = logging.getLogger("collector.load_provided")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = PROJECT_ROOT / "data" / "provided"
DEFAULT_DB = storage.DEFAULT_DB
DEFAULT_GLOB = "04_바이오헬스_*.csv"

# --------------------------------------------------------------------------- #
# 스키마
# --------------------------------------------------------------------------- #
# 제공 CSV 원본 컬럼 → 내부 표준 컬럼
COLUMN_MAP = {
    "제목": "title",
    "본문전문": "body",
    "원문URL": "url",
    "발행일": "published_at",
    "매체명": "press",
    "문서종류": "doc_type",
    "수집키워드": "collect_keywords",
}


# 수집 키워드별 본문 매칭 패턴 (띄어쓰기 변형 허용)
CORE_KEYWORDS = {
    "바이오 의약품": r"바이오\s*의약품",
    "디지털 헬스케어": r"디지털\s*헬스\s*케어",
}

# 과제 안내의 '기술 하위어 / 시장·정책' 키워드 + EDA에서 빈도가 높았던 하위 주제
TOPIC_PATTERNS = {
    "ADC": r"(?<![A-Za-z])ADC(?![A-Za-z])|항체\s*-?\s*약물\s*접합",
    "CDMO": r"(?<![A-Za-z])C(?:D)?MO(?![A-Za-z])|위탁\s*(?:개발|생산)",
    "AI 신약개발": r"(?:AI|인공지능)\s*(?:기반\s*)?신약",
    "바이오 투자": r"투자\s*유치|시리즈\s*[A-D](?![A-Za-z])|벤처\s*캐피(?:탈|털)|(?<![A-Za-z])VC(?![A-Za-z])|(?<![A-Za-z])IPO(?![A-Za-z])|기업\s*공개|상장\s*(?:예정|추진|심사)",
    "신약 기술수출": r"기술\s*수출|라이선스\s*아웃|(?<![A-Za-z])L/O(?![A-Za-z])|기술\s*이전\s*계약",
    "의료 AI 규제": r"(?:의료|헬스케어)\s*AI.{0,40}(?:규제|허가|인증|가이드라인)|규제\s*샌드박스|디지털\s*의료\s*제품법|혁신\s*의료\s*기기",
    "비대면 진료": r"비대면\s*진료|원격\s*(?:의료|진료)",
    "바이오시밀러": r"바이오\s*시밀러",
}

# 시장 전망성 서술 여부 (뉴스 안의 '전망' 문장 → 보고서와 상충 비교 후보)
OUTLOOK_PATTERN = r"전망|시장\s*규모|연평균\s*성장률|CAGR|억\s*달러\s*규모|성장할\s*것"

# 포털 수집 과정에서 붙은 보일러플레이트
BOILERPLATE_PATTERNS = [
    r"기사를\s*더\s*보고\s*싶다면\?",
    r"기사\s*섹션\s*분류\s*안내",
    r"이\s*기사를\s*추천합니다",
    r"\(?[^()\s]*\s*재판매\s*및\s*DB\s*금지\)?",
    r"Copyright\s*(?:ⓒ|\(c\))?.{0,40}?All\s*rights\s*reserved\.?",
    r"무단\s*전재[^.]{0,40}금지\.?",
]

# 정보량이 적은 기사 형식 (제목 접두어)
# ([게시판]류 단신 모음은 투자·제휴 정보가 많아 유지)
NOISE_TITLE_PATTERN = r"^\s*[\[\(<]?\s*(?:[^\]\)>]{0,6}포토|사진|인사|부고|동정|화촉|알림)\s*[\]\)>]"

# 산업 도메인 일반어 — 핵심 키워드가 없더라도 바이오·헬스 기사인지 판단하는 보조 신호
DOMAIN_PATTERN = r"바이오|제약|의약|헬스|의료|신약|병원|임상|치료제|진단|백신|항체|세포|유전자"
MIN_DOMAIN_HITS = 3

# 뉴스 본문이 아닌 섹션 (연예·스포츠)
NON_NEWS_HOSTS = ("entertain.naver.com", "sports.naver.com", "sports.news.naver.com")

MIN_BODY_CHARS = 300  # 정제 후 본문 최소 길이 (사진 캡션·속보 한 줄 제거)


@dataclass
class LoadStats:
    files: dict[str, int] = field(default_factory=dict)
    raw_rows: int = 0
    unique_docs: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    selected: int = 0
    topics: dict[str, int] = field(default_factory=dict)
    period: tuple[str, str] | None = None

    def to_dict(self) -> dict:
        return self.__dict__ | {"generated_at": datetime.now().isoformat(timespec="seconds")}


# --------------------------------------------------------------------------- #
# 1) 적재
# --------------------------------------------------------------------------- #
def nfc(s: str) -> str:
    """macOS에서 만든 파일명은 NFD(자모 분리)라 NFC로 정규화한다."""
    return unicodedata.normalize("NFC", s)


def discover_files(src: Path, pattern: str = DEFAULT_GLOB) -> list[Path]:
    if src.is_file():
        return [src]
    want = nfc(pattern)
    files = sorted(p for p in src.glob("*.csv") if Path(nfc(p.name)).match(want))
    if not files:
        raise FileNotFoundError(f"{src} 에서 '{pattern}' 패턴의 CSV를 찾지 못했습니다.")
    return files


def read_provided(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: 필수 컬럼 누락 {sorted(missing)}")
    df = df[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["source_file"] = nfc(path.name)
    return df


def make_doc_id(url: str) -> str:
    """네이버 뉴스 URL(여러 형식)에서 언론사ID(oid)·기사ID(aid)를 뽑아 안정적인 ID를 만든다."""
    m = re.search(r"/article/(?:comment/)?(\d{3})/(\d{10})", url) or re.search(
        r"oid=(\d{3}).*?aid=(\d{10})", url
    )
    if m:
        return f"naver:{m.group(1)}:{m.group(2)}"
    return "url:" + re.sub(r"[?#].*$", "", url.strip())


def split_keywords(s: str) -> set[str]:
    return {k.strip() for k in str(s).split(";") if k.strip()}


def merge_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """파일 간 중복(묶음 파일 ⊇ 키워드 파일)을 doc_id 기준으로 합치고 수집키워드를 합집합으로 보존."""
    df = pd.concat(frames, ignore_index=True)
    df["doc_id"] = df["url"].map(make_doc_id)
    df["_kw"] = df["collect_keywords"].map(split_keywords)
    df["_len"] = df["body"].str.len()

    agg_kw = df.groupby("doc_id")["_kw"].agg(lambda s: set().union(*s))
    agg_src = df.groupby("doc_id")["source_file"].agg(lambda s: sorted(set(s)))
    # 같은 doc_id 안에서는 본문이 가장 긴 행을 대표로
    rep = df.sort_values("_len", ascending=False).drop_duplicates("doc_id").set_index("doc_id")
    rep["collect_keywords"] = agg_kw.map(lambda k: "; ".join(sorted(k)))
    rep["source_files"] = agg_src.map(lambda s: "; ".join(s))
    return rep.drop(columns=["_kw", "_len", "source_file"]).reset_index()


# --------------------------------------------------------------------------- #
# 2) 정제
# --------------------------------------------------------------------------- #
_BOILER_RE = re.compile("|".join(BOILERPLATE_PATTERNS), flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    if not isinstance(text, str) or text.lower() == "nan":
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _BOILER_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 꼬리 보일러플레이트를 지운 뒤에야 끝에 드러나는 저작권 표기
    return re.sub(r"\s*Copyrights?$", "", text)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].map(clean_text)
    df["body"] = df["body"].map(clean_text)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df["doc_type"] = df["doc_type"].str.strip().str.lower().replace("", "news")
    df["host"] = df["url"].str.extract(r"https?://([^/]+)", expand=False).fillna("")
    df["char_len"] = df["body"].str.len()
    return df


# --------------------------------------------------------------------------- #
# 3) 선별
# --------------------------------------------------------------------------- #
def keyword_hits(row: pd.Series) -> int:
    text = f"{row['title']} {row['body']}"
    pats = [CORE_KEYWORDS[k] for k in split_keywords(row["collect_keywords"]) if k in CORE_KEYWORDS]
    pats = pats or list(CORE_KEYWORDS.values())
    return sum(len(re.findall(p, text)) for p in pats)


def tag_topics(text: str) -> list[str]:
    return [name for name, pat in TOPIC_PATTERNS.items() if re.search(pat, text)]


def select(df: pd.DataFrame, since: str | None = None, until: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """규칙 기반 선별. 먼저 걸린 규칙 하나를 drop_reason으로 기록한다."""
    df = df.copy()
    full = df["title"] + " " + df["body"]
    df["keyword_hits"] = df.apply(keyword_hits, axis=1)
    df["topics"] = full.map(lambda t: "; ".join(tag_topics(t)))
    df["is_outlook"] = full.str.contains(OUTLOOK_PATTERN, regex=True).astype(int)
    df["domain_hits"] = full.str.count(DOMAIN_PATTERN)

    reason = pd.Series("", index=df.index, dtype=object)

    def mark(mask: pd.Series, why: str) -> None:
        reason[(reason == "") & mask] = why

    mark(df["published_at"].isna(), "invalid_date")
    if since:
        mark(df["published_at"] < pd.Timestamp(since), "out_of_period")
    if until:
        mark(df["published_at"] > pd.Timestamp(until), "out_of_period")
    mark(df["host"].str.endswith(NON_NEWS_HOSTS), "non_news_section")
    mark(df["body"] == "", "empty_body")
    mark(df["title"].str.contains(NOISE_TITLE_PATTERN, regex=True), "noise_title")
    mark(df["char_len"] < MIN_BODY_CHARS, "short_body")
    # 핵심 키워드·하위 주제가 없고 도메인 일반어도 드물면 주제 무관(검색어가 메뉴·추천기사에서만 걸린 경우)
    mark(
        (df["keyword_hits"] == 0) & (df["topics"] == "") & (df["domain_hits"] < MIN_DOMAIN_HITS),
        "off_topic",
    )
    # 본문 완전 중복(통신사 전재, 동일 기사 제목만 수정 재송고) → 가장 이른 기사만 유지
    alive = df[reason == ""].sort_values("published_at")
    dup = alive["body"].duplicated(keep="first").reindex(df.index, fill_value=False)
    mark(dup, "exact_dup_body")

    df["drop_reason"] = reason
    kept = df[reason == ""].drop(columns="drop_reason")
    rejected = df[reason != ""]
    return kept, rejected


# --------------------------------------------------------------------------- #
# 4) 저장
# --------------------------------------------------------------------------- #
def to_records(df: pd.DataFrame, loaded_at: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "doc_id": df["doc_id"],
            "source_type": df["doc_type"],
            "evidence_role": df["doc_type"].map(EVIDENCE_ROLE).fillna("current"),
            "origin": "provided",
            "title": df["title"],
            "body": df["body"],
            "url": df["url"],
            "published_at": df["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "publisher": df["press"],
            "collect_keywords": df["collect_keywords"],
            "topics": df["topics"],
            "is_outlook": df["is_outlook"],
            "keyword_hits": df["keyword_hits"],
            "char_len": df["char_len"],
            "source_files": df["source_files"],
            "loaded_at": loaded_at,
        }
    )
    return out


def save(kept: pd.DataFrame, rejected: pd.DataFrame, db_path: Path) -> None:
    loaded_at = storage.now_iso()
    docs = to_records(kept, loaded_at)
    rej = pd.DataFrame(
        {
            "doc_id": rejected["doc_id"],
            "title": rejected["title"],
            "url": rejected["url"],
            "published_at": rejected["published_at"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "drop_reason": rejected["drop_reason"],
            "char_len": rejected["char_len"],
            "loaded_at": loaded_at,
        }
    )
    # 재실행 시 제공 데이터분만 교체 (크롤링 문서는 유지)
    storage.save_documents(docs, "provided", db_path, rejected=rej, replace_origin=True)
    log.info("저장 완료: %s (documents %d건, rejected_docs %d건)", db_path, len(docs), len(rej))


# --------------------------------------------------------------------------- #
# 파이프라인
# --------------------------------------------------------------------------- #
def run(
    src: Path = DEFAULT_SRC,
    db_path: Path = DEFAULT_DB,
    pattern: str = DEFAULT_GLOB,
    since: str | None = None,
    until: str | None = None,
    dry_run: bool = False,
) -> LoadStats:
    stats = LoadStats()
    frames = []
    for path in discover_files(src, pattern):
        df = read_provided(path)
        stats.files[nfc(path.name)] = len(df)
        frames.append(df)
        log.info("읽음: %s (%d행)", nfc(path.name), len(df))
    stats.raw_rows = sum(stats.files.values())

    merged = normalize(merge_sources(frames))
    stats.unique_docs = len(merged)
    log.info("병합: %d행 → 고유 문서 %d건", stats.raw_rows, stats.unique_docs)

    kept, rejected = select(merged, since, until)
    stats.selected = len(kept)
    stats.dropped = rejected["drop_reason"].value_counts().to_dict()
    stats.topics = (
        kept["topics"].str.split("; ").explode().loc[lambda s: s != ""].value_counts().to_dict()
    )
    if len(kept):
        stats.period = (str(kept["published_at"].min()), str(kept["published_at"].max()))
    log.info("선별: %d건 유지, 제외 %s", stats.selected, stats.dropped)

    if not dry_run:
        save(kept, rejected, db_path)
        report = db_path.parent / "load_provided_stats.json"
        report.write_text(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="제공 CSV 적재 및 선별 → store/docs.db")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="CSV 파일 또는 디렉터리")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--pattern", default=DEFAULT_GLOB, help="디렉터리 내 CSV 파일명 패턴")
    ap.add_argument("--since", help="이 날짜 이후 기사만 (YYYY-MM-DD)")
    ap.add_argument("--until", help="이 날짜 이전 기사만 (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true", help="DB에 쓰지 않고 통계만 출력")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run(args.src, args.db, args.pattern, args.since, args.until, args.dry_run)
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

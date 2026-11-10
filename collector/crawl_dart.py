"""① 수집 계층 — DART(전자공시) OpenAPI 수집.

바이오·헬스케어 상장사의 공시 중 주제 B에 의미 있는 것만 골라 원문을 받아 store/docs.db에 적재한다.

  1) 기업 고유번호(corpCode.xml) 내려받기 → 대상 기업 유니버스 구성
     - 기본: SEED_COMPANIES (세그먼트별 대표 기업)
     - --expand-ksic: 상장사 전체의 업종코드(company.json)를 조회해 의약품·의료기기·의약 R&D 업종을 추가
  2) 기업별 공시 목록(list.json) 조회 → 공시명 패턴으로 선별 (REPORT_RULES)
  3) 공시 원문(document.xml) 내려받기 → 텍스트 변환
     - 거래소 수시공시(HTML 표)는 '항목 | 값' 줄로 변환
     - 사업보고서(dart4 XML)는 'II. 사업의 내용'만 추출 (지식베이스 규모 조정: 재무·경영진단 장 제외)
  4) documents 테이블에 source_type='disclosure'로 upsert (이미 적재된 접수번호는 건너뜀)

원문 zip은 data/dart/raw/ 에 캐시하므로 재실행 시 API를 다시 부르지 않는다.

실행 예:
    python -m collector.crawl_dart                       # 시드 기업, 2021-10-16 ~ 오늘
    python -m collector.crawl_dart --companies 알테오젠 리가켐바이오 --since 2024-01-01
    python -m collector.crawl_dart --expand-ksic --no-annual
    python -m collector.crawl_dart --dry-run             # 목록·선별만 확인 (원문 X, DB X)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from dotenv import load_dotenv

from collector import storage
from collector.load_provided import OUTLOOK_PATTERN, TOPIC_PATTERNS

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
log = logging.getLogger("collector.crawl_dart")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "dart"
API = "https://opendart.fss.or.kr/api/"
VIEWER = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={}"
ORIGIN = "dart"
DEFAULT_SINCE = "2021-10-16"  # 제공 뉴스 데이터와 같은 기간
REQUEST_INTERVAL = 0.15       # 초당 요청 수 제한 (과도한 호출 시 DART가 키를 차단함)

# --------------------------------------------------------------------------- #
# 대상 기업
# --------------------------------------------------------------------------- #
# 과제 키워드표의 하위 주제별 대표 상장사 (DART 등록 회사명 기준)
SEED_COMPANIES: dict[str, list[str]] = {
    "CDMO·바이오의약품": [
        "삼성바이오로직스", "셀트리온", "SK바이오사이언스", "에스티팜", "바이넥스",
        "프레스티지바이오로직스", "녹십자", "휴온스",
    ],
    "ADC·플랫폼 기술수출": [
        "알테오젠", "리가켐바이오", "에이비엘바이오", "오름테라퓨틱", "인투셀",
        "펩트론", "디앤디파마텍", "보로노이", "올릭스",
    ],
    "제약·신약": [
        "유한양행", "한미약품", "대웅제약", "종근당", "동아에스티", "에스케이바이오팜",
        "HLB", "삼천당제약", "JW중외제약", "일동제약", "보령",
    ],
    "AI 신약개발": ["신테카바이오", "파로스아이바이오", "온코크로스"],
    "디지털 헬스케어·의료 AI": [
        "루닛", "뷰노", "제이엘케이", "딥노이드", "코어라인소프트",
        "인바디", "비트컴퓨터", "셀바스헬스케어",
    ],
}

# --expand-ksic: 표준산업분류 코드 접두어
KSIC_PREFIXES = {
    "21": "의료용 물질 및 의약품 제조업",
    "271": "의료용 기기 제조업",
    "70113": "의학 및 약학 연구개발업",
}

# --------------------------------------------------------------------------- #
# 공시 선별 규칙 (공시명 기준, 위에서부터 먼저 맞는 규칙 적용)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReportRule:
    category: str
    pattern: str
    evidence_role: str = "current"
    topics: tuple[str, ...] = ()


REPORT_RULES = [
    ReportRule("실적전망", r"영업실적등에대한전망", "outlook"),
    ReportRule("기술이전·라이선스", r"기술(?:도입|이전|수출)|라이선스|라이센스|마일스톤|기술료", topics=("신약 기술수출",)),
    ReportRule("투자판단 주요경영사항", r"투자판단관련주요경영사항"),  # 임상·허가·계약 등 (괄호 안 상세 제목)
    ReportRule("공급계약", r"단일판매.?공급계약"),
    ReportRule("설비투자", r"신규시설투자|유형자산취득결정", topics=("CDMO",)),
    ReportRule("자금조달", r"유상증자결정|전환사채권발행결정|신주인수권부사채권발행결정", topics=("바이오 투자",)),
    ReportRule("타법인 투자·M&A", r"타법인주식및출자증권(?:취득|양수)결정|회사합병결정|영업양수결정"),
    ReportRule("사업보고서", r"^사업보고서"),
]
SKIP_PREFIX = re.compile(r"^\[(?:첨부정정|첨부추가|발행조건확정)\]")
ANNUAL_SECTIONS = ("사업의 내용",)


class DartError(RuntimeError):
    pass


class DartLimitExceeded(DartError):
    pass


# --------------------------------------------------------------------------- #
# API 클라이언트
# --------------------------------------------------------------------------- #
class DartClient:
    def __init__(self, api_key: str, interval: float = REQUEST_INTERVAL):
        self.key = api_key
        self.interval = interval
        self.session = requests.Session()
        self._last = 0.0
        self.calls = 0

    def _get(self, endpoint: str, **params) -> requests.Response:
        wait = self.interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                resp = self.session.get(API + endpoint, params={"crtfc_key": self.key, **params}, timeout=120)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise DartError(f"{endpoint} 요청 실패: {e}") from e
                time.sleep(2 ** attempt)
        self._last = time.monotonic()
        self.calls += 1
        return resp

    @staticmethod
    def _check(status: str, message: str) -> None:
        if status in ("000", "013"):  # 013 = 조회된 데이터 없음
            return
        if status == "020":
            raise DartLimitExceeded(f"DART 요청 한도 초과: {message}")
        raise DartError(f"DART 오류 {status}: {message}")

    def json(self, endpoint: str, **params) -> dict:
        data = self._get(endpoint, **params).json()
        self._check(data.get("status", ""), data.get("message", ""))
        return data

    def zip_files(self, endpoint: str, **params) -> dict[str, bytes]:
        content = self._get(endpoint, **params).content
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                return {n: z.read(n) for n in z.namelist()}
        except zipfile.BadZipFile:
            # 오류 시 zip 대신 XML/JSON 상태 메시지가 온다
            m = re.search(rb"<status>(\d+)</status>.*?<message>(.*?)</message>", content, re.S)
            if m:
                self._check(m.group(1).decode(), m.group(2).decode("utf-8", "replace"))
            raise DartError(f"{endpoint}: zip이 아닌 응답 {content[:200]!r}")

    # --- endpoints ---------------------------------------------------------
    def corp_codes(self, cache: Path) -> pd.DataFrame:
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            files = self.zip_files("corpCode.xml")
            cache.write_bytes(next(iter(files.values())))
        root = ET.fromstring(cache.read_bytes())
        rows = [{c.tag: (c.text or "").strip() for c in e} for e in root.iter("list")]
        return pd.DataFrame(rows)

    def company(self, corp_code: str) -> dict:
        return self.json("company.json", corp_code=corp_code)

    def disclosures(self, corp_code: str, since: str, until: str) -> list[dict]:
        out, page = [], 1
        while True:
            data = self.json(
                "list.json", corp_code=corp_code, bgn_de=since, end_de=until,
                page_no=page, page_count=100,
            )
            out += data.get("list", [])
            if page >= int(data.get("total_page", 0) or 0):
                return out
            page += 1

    def document(self, rcept_no: str, cache_dir: Path) -> dict[str, bytes]:
        """원문 zip 내용. data/dart/raw/{rcept_no}.zip 에 캐시."""
        path = cache_dir / f"{rcept_no}.zip"
        if path.exists():
            with zipfile.ZipFile(path) as z:
                return {n: z.read(n) for n in z.namelist()}
        files = self.zip_files("document.xml", rcept_no=rcept_no)
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for n, b in files.items():
                z.writestr(n, b)
        return files


# --------------------------------------------------------------------------- #
# 유니버스
# --------------------------------------------------------------------------- #
def build_universe(client: DartClient, names: list[str] | None, expand_ksic: bool) -> pd.DataFrame:
    corps = client.corp_codes(CACHE_DIR / "corpCode.xml")
    listed = corps[corps["stock_code"] != ""].drop_duplicates("corp_code")

    if names:
        targets = {n: "사용자 지정" for n in names}
    else:
        targets = {n: seg for seg, ns in SEED_COMPANIES.items() for n in ns}

    by_name = listed.set_index("corp_name")
    rows, missing = [], []
    for name, seg in targets.items():
        if name in by_name.index:
            r = by_name.loc[[name]].iloc[0]
            rows.append({"corp_code": r["corp_code"], "corp_name": name, "stock_code": r["stock_code"], "segment": seg})
        else:
            missing.append(name)
    if missing:
        log.warning("DART 상장사 목록에서 찾지 못한 회사(건너뜀): %s", ", ".join(missing))

    if expand_ksic:
        rows += ksic_companies(client, listed)

    uni = pd.DataFrame(rows).drop_duplicates("corp_code")
    log.info("대상 기업 %d곳", len(uni))
    return uni


def ksic_companies(client: DartClient, listed: pd.DataFrame) -> list[dict]:
    """상장사 업종코드를 조회(캐시)해 바이오·헬스케어 업종만 반환. 최초 1회 약 4천 회 호출."""
    cache = CACHE_DIR / "company_info.json"
    info: dict[str, dict] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [c for c in listed["corp_code"] if c not in info]
    if todo:
        log.info("업종코드 조회: %d곳 (캐시 %d곳)", len(todo), len(info))
    try:
        for i, code in enumerate(todo, 1):
            d = client.company(code)
            info[code] = {k: d.get(k, "") for k in ("corp_name", "stock_code", "induty_code", "corp_cls")}
            if i % 200 == 0:
                cache.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
                log.info("  … %d/%d", i, len(todo))
    finally:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

    names = dict(zip(listed["corp_code"], listed["corp_name"]))
    out = []
    for code, d in info.items():
        if code not in names:
            continue
        for prefix, label in KSIC_PREFIXES.items():
            if d.get("induty_code", "").startswith(prefix):
                out.append({"corp_code": code, "corp_name": names[code], "stock_code": d["stock_code"], "segment": f"KSIC {label}"})
                break
    log.info("업종코드 기준 바이오·헬스케어 상장사 %d곳", len(out))
    return out


# --------------------------------------------------------------------------- #
# 공시 선별
# --------------------------------------------------------------------------- #
def classify(report_nm: str, with_annual: bool = True) -> ReportRule | None:
    name = re.sub(r"\s+", "", report_nm)
    if SKIP_PREFIX.match(name):
        return None
    base = re.sub(r"^\[[^\]]+\]", "", name)  # [기재정정] 등 접두어 제거 후 판정
    for rule in REPORT_RULES:
        if re.search(rule.pattern, base):
            if rule.category == "사업보고서" and not with_annual:
                return None
            return rule
    return None


# --------------------------------------------------------------------------- #
# 원문 → 텍스트
# --------------------------------------------------------------------------- #
def _html_to_text(markup: str) -> str:
    soup = BeautifulSoup(markup, "lxml")
    for t in soup(["style", "script", "head"]):
        t.decompose()
    for t in soup.find_all("title"):  # dart4 XML의 섹션 제목
        t.replace_with(f"\n## {t.get_text(' ', strip=True)}\n")
    for table in soup.find_all("table"):
        lines = []
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th", "te", "tu"])]
            cells = [c for c in cells if c]
            if cells:
                lines.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(lines) + "\n")
    text = soup.get_text("\n")
    text = re.sub(r"[ \t 　]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _decode(b: bytes) -> str:
    for enc in ("utf-8", "cp949"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode("utf-8", "replace")


def extract_annual_sections(xml: str, wanted: tuple[str, ...] = ANNUAL_SECTIONS) -> str:
    """사업보고서 본문에서 SECTION-1 단위로 원하는 장만 잘라낸다 (재무제표 주석 등 수 MB 제외)."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"<SECTION-1[^>]*>\s*<TITLE[^>]*>([^<]*)</TITLE>", xml)]
    parts = []
    for i, (pos, title) in enumerate(starts):
        if any(w in title for w in wanted):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(xml)
            parts.append(xml[pos:end])
    return "\n".join(parts)


def document_text(files: dict[str, bytes], rcept_no: str, annual: bool) -> str:
    main = files.get(f"{rcept_no}.xml") or next(iter(files.values()))
    markup = _decode(main)
    if annual:
        markup = extract_annual_sections(markup) or markup[:300_000]
    return _html_to_text(markup)


def detail_title(report_nm: str) -> str:
    """'투자판단관련주요경영사항 (ALT-B4 … 계약 체결)' → 괄호 안 상세 제목.
    '사업보고서 (2025.12)'처럼 괄호가 기간이면 상세 제목이 없는 것으로 본다."""
    # 마지막 괄호 묶음 (한 단계 중첩 허용): '…(임상시험계획승인신청등결정) (안드로겐성 …)' → 뒤쪽
    m = re.search(r"\(((?:[^()]|\([^()]*\))+)\)\s*$", report_nm.strip())
    if not m or re.fullmatch(r"\d{4}\.\d{2}|(?:자율|공정|안내)공시|종속회사의주요경영사항", m.group(1).strip()):
        return ""
    return m.group(1).strip()


def body_title(body: str) -> str:
    """공시 본문 서식에서 제목을 찾는다: '1. 제목 | …' 또는 '체결계약명 | …' (+ 계약상대)."""
    m = re.search(r"^\s*(?:\d+\.\s*)?제목\s*\|\s*(.+)$", body, re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r"^\s*-?\s*(?:\d+\.\s*)?(?:체결계약명|판매ㆍ공급계약\s*내용)\s*\|\s*(.+)$", body, re.M)
    if m:
        party = re.search(r"^\s*(?:\d+\.\s*)?계약상대(?:방)?\s*\|\s*(.+)$", body, re.M)
        return m.group(1).strip() + (f" (계약상대: {party.group(1).strip()})" if party else "")
    return ""


def latest_annual_only(items: list[dict]) -> list[dict]:
    """같은 기간 사업보고서의 원본·[기재정정]본 중 가장 최근 접수본만 남긴다."""
    latest: dict[str, str] = {}
    for it in items:
        name = re.sub(r"\s+", "", it["report_nm"])
        base = re.sub(r"^\[[^\]]+\]", "", name)
        if base.startswith("사업보고서"):
            latest[base] = max(latest.get(base, ""), it["rcept_no"])
    keep = set(latest.values())
    return [
        it for it in items
        if not re.sub(r"^\[[^\]]+\]", "", re.sub(r"\s+", "", it["report_nm"])).startswith("사업보고서")
        or it["rcept_no"] in keep
    ]


# --------------------------------------------------------------------------- #
# 파이프라인
# --------------------------------------------------------------------------- #
def collect(
    client: DartClient,
    universe: pd.DataFrame,
    since: str,
    until: str,
    with_annual: bool = True,
    skip_ids: set[str] | None = None,
    max_docs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(선별된 공시 목록, 원문 포함 문서) 반환. 한도 초과 시 그때까지 모은 것만 반환."""
    skip_ids = skip_ids or set()
    picked, docs = [], []
    raw_dir = CACHE_DIR / "raw"
    try:
        for _, corp in universe.iterrows():
            items = latest_annual_only(client.disclosures(corp["corp_code"], since, until))
            n_pick = 0
            for it in items:
                rule = classify(it["report_nm"], with_annual)
                if rule is None:
                    continue
                n_pick += 1
                picked.append({**it, "category": rule.category, "segment": corp["segment"]})
                doc_id = f"dart:{it['rcept_no']}"
                if doc_id in skip_ids or (max_docs is not None and len(docs) >= max_docs):
                    continue
                try:
                    files = client.document(it["rcept_no"], raw_dir)
                    body = document_text(files, it["rcept_no"], annual=rule.category == "사업보고서")
                except DartLimitExceeded:
                    raise
                except DartError as e:
                    log.warning("원문 실패 %s %s: %s", corp["corp_name"], it["rcept_no"], e)
                    continue
                docs.append(to_document(it, rule, corp, body))
            log.info("%-12s 공시 %4d건 → 선별 %3d건", corp["corp_name"], len(items), n_pick)
    except DartLimitExceeded as e:
        log.error("%s — 지금까지 수집한 %d건만 저장합니다. 내일 다시 실행하면 이어서 수집합니다.", e, len(docs))
    return pd.DataFrame(picked), pd.DataFrame(docs)


def to_document(it: dict, rule: ReportRule, corp: pd.Series, body: str) -> dict:
    report_nm = re.sub(r"\s+", " ", it["report_nm"]).strip()
    detail = detail_title(report_nm) or body_title(body)
    prefix = "[정정] " if re.match(r"^\[(?:기재)?정정\]", report_nm) else ""
    base = re.sub(r"^\[[^\]]+\]\s*", "", report_nm)
    title = f"[{it['corp_name']}] {prefix}{detail or base}"
    full = f"{title}\n{body}"
    topics = [t for t, p in TOPIC_PATTERNS.items() if re.search(p, full)]
    topics += [t for t in rule.topics if t not in topics]
    rcept_dt = it["rcept_dt"]
    return {
        "doc_id": f"dart:{it['rcept_no']}",
        "source_type": "disclosure",
        "evidence_role": rule.evidence_role,
        "title": title,
        "body": body,
        "url": VIEWER.format(it["rcept_no"]),
        "published_at": f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}T00:00:00",
        "publisher": it["corp_name"],
        "collect_keywords": rule.category,
        "topics": "; ".join(topics),
        "is_outlook": int(rule.evidence_role == "outlook" or bool(re.search(OUTLOOK_PATTERN, body))),
        "keyword_hits": None,
        "char_len": len(body),
        "source_files": None,
        "meta": json.dumps(
            {
                "rcept_no": it["rcept_no"],
                "corp_code": it["corp_code"],
                "stock_code": it.get("stock_code", ""),
                "corp_cls": it.get("corp_cls", ""),
                "report_nm": report_nm,
                "category": rule.category,
                "segment": corp["segment"],
                "remark": it.get("rm", ""),  # 유=유가증권 코=코스닥 정=정정 연=연결 …
            },
            ensure_ascii=False,
        ),
    }


def run(
    names: list[str] | None = None,
    since: str = DEFAULT_SINCE,
    until: str | None = None,
    expand_ksic: bool = False,
    with_annual: bool = True,
    refresh: bool = False,
    max_docs: int | None = None,
    dry_run: bool = False,
    db_path: Path = storage.DEFAULT_DB,
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("DART_API_KEY")
    if not key:
        raise SystemExit("DART_API_KEY가 없습니다. .env에 추가하세요 (.env.example 참고).")
    client = DartClient(key)

    since_ = since.replace("-", "")
    until_ = (until or date.today().isoformat()).replace("-", "")
    universe = build_universe(client, names, expand_ksic)

    if dry_run:
        picked = []
        for _, corp in universe.iterrows():
            for it in latest_annual_only(client.disclosures(corp["corp_code"], since_, until_)):
                rule = classify(it["report_nm"], with_annual)
                if rule:
                    picked.append({"corp_name": it["corp_name"], "category": rule.category, "report_nm": it["report_nm"].strip()})
        df = pd.DataFrame(picked)
        stats = {
            "companies": len(universe),
            "selected": len(df),
            "by_category": df["category"].value_counts().to_dict() if len(df) else {},
            "by_company": df["corp_name"].value_counts().to_dict() if len(df) else {},
            "api_calls": client.calls,
        }
        return stats

    skip = set() if refresh else storage.existing_ids(ORIGIN, db_path)
    picked, docs = collect(client, universe, since_, until_, with_annual, skip, max_docs)
    if len(docs):
        storage.save_documents(docs, ORIGIN, db_path)
    stats = {
        "companies": len(universe),
        "selected_disclosures": len(picked),
        "already_loaded": 0 if picked.empty else int(picked["rcept_no"].map(lambda r: f"dart:{r}" in skip).sum()),
        "saved": len(docs),
        "by_category": docs["collect_keywords"].value_counts().to_dict() if len(docs) else {},
        "by_role": docs["evidence_role"].value_counts().to_dict() if len(docs) else {},
        "topics": docs["topics"].str.split("; ").explode().loc[lambda s: s.fillna("") != ""].value_counts().to_dict() if len(docs) else {},
        "api_calls": client.calls,
        "generated_at": storage.now_iso(),
    }
    (db_path.parent / "crawl_dart_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="DART OpenAPI로 바이오·헬스케어 상장사 공시 수집 → store/docs.db")
    ap.add_argument("--companies", nargs="+", help="회사명 직접 지정 (기본: 시드 기업 목록)")
    ap.add_argument("--since", default=DEFAULT_SINCE, help="시작일 YYYY-MM-DD")
    ap.add_argument("--until", help="종료일 YYYY-MM-DD (기본: 오늘)")
    ap.add_argument("--expand-ksic", action="store_true", help="업종코드로 바이오·헬스케어 상장사 전체 추가")
    ap.add_argument("--no-annual", action="store_true", help="사업보고서 제외")
    ap.add_argument("--refresh", action="store_true", help="이미 적재된 공시도 다시 처리")
    ap.add_argument("--max-docs", type=int, help="이번 실행에서 원문을 받을 최대 건수 (테스트용)")
    ap.add_argument("--dry-run", action="store_true", help="목록 조회·선별만 (원문 다운로드·DB 저장 없음)")
    ap.add_argument("--db", type=Path, default=storage.DEFAULT_DB)
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = run(
        args.companies, args.since, args.until, args.expand_ksic, not args.no_annual,
        args.refresh, args.max_docs, args.dry_run, args.db,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

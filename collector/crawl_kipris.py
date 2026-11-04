"""① 수집 계층 — KIPRIS Plus OpenAPI 특허 수집 (기술 선행 지표, evidence_role='leading').

신청 서비스 (KIPRIS Plus 'API' 목록 번호):
   1  특허·실용 공개·등록공보   patUtiModInfoSearchSevice   검색·서지·초록·청구항
  34  대표 출원인               RpstApplicantService        출원인 명칭 정규화 → DART 기업과 연결
  27  특허 패밀리               patFamInfoSearchService     해외 출원국 → 글로벌 진출(기술수출 선행 신호)
  25  특허·실용 피인용문헌      CitingService               피인용 수 → 핵심 특허 가중치
  48  해외특허                  ForeignPatent*Service       미국·유럽·PCT 동향 비교

수집 방식
  - 주제별 IPC/CPC 검색 (TOPIC_QUERIES): ADC, AI 신약개발, 디지털 헬스케어·의료 AI, CDMO·바이오의약품
  - 기업별 출원인 검색 (--by-applicant): DART 시드 기업 → 공시와 특허를 같은 회사로 연결
  - 문서 본문 = 발명의 명칭 + 출원인 + IPC + 초록 (+ 대표 청구항)
  - API 응답은 data/kipris/cache/ 에 캐시 (재실행 시 호출 수 절약)

실행 예:
    python -m collector.crawl_kipris --check                  # 키·서비스별 활성 상태 확인
    python -m collector.crawl_kipris                          # 주제별 국내 특허
    python -m collector.crawl_kipris --by-applicant           # DART 시드 기업 특허
    python -m collector.crawl_kipris --foreign US EP          # 해외 특허 (주제별 IPC)
    python -m collector.crawl_kipris --enrich family citing --claims --budget 300   # 특허별 보강 (건당 +3회)

무료 이용은 월 1,000회 한도라 기본값은 '검색만'이다. 국내 검색은 1회 호출에 최대 500건(서지+초록)을
받으므로 주제별 전체를 수십 회 안에 수집할 수 있다. 청구항·대표출원인·패밀리·피인용은 특허마다
1회씩 추가로 드니 필요한 특허에만 --claims / --enrich 로 켠다. --budget 으로 실행당 호출 상한을 건다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from collector import storage
from collector.load_provided import TOPIC_PATTERNS

log = logging.getLogger("collector.crawl_kipris")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "kipris" / "cache"
BASE = "http://plus.kipris.or.kr/"
ORIGIN = "kipris"
DEFAULT_SINCE = "2021-10-16"
# 국내 공보 열람 링크 (KIPRIS DOI → kipris.or.kr 상세화면). kipris.or.kr은 robots.txt로 수집을 막으므로
# 링크는 출처 표기용으로만 저장하고, 데이터는 KIPRIS Plus OpenAPI로만 받는다.
KR_DOI = "https://doi.org/10.8080/{}"


def api_url(name: str, **params) -> str:
    """출처 기록용 API 요청 URL (인증키 제외)."""
    from urllib.parse import urlencode
    return BASE + EP[name][0] + "?" + urlencode(params)
REQUEST_INTERVAL = 0.2

# --------------------------------------------------------------------------- #
# 엔드포인트 (KIPRIS Plus 명세 기준. kipo-api 계열은 ServiceKey, openapi/rest 계열은 accessKey)
# --------------------------------------------------------------------------- #
EP = {
    "search": ("kipo-api/kipi/patUtiModInfoSearchSevice/getAdvancedSearch", "ServiceKey"),
    "claim": ("openapi/rest/patUtiModInfoSearchSevice/patentClaimInfo", "accessKey"),
    "rpst_applicant": ("openapi/rest/RpstApplicantService/rpstApplicantInfoV2", "accessKey"),
    "family": ("kipo-api/kipi/patFamInfoSearchService/getAppNoPatFamInfoSearch", "ServiceKey"),
    "citing": ("openapi/rest/CitingService/citingInfo", "accessKey"),
    "foreign_search": ("openapi/rest/ForeignPatentAdvencedSearchService/advancedSearch", "accessKey"),
    "foreign_abstract": ("openapi/rest/ForeignPatentBibliographicService/summation", "accessKey"),
}

# --check 에서 쓰는 명세 샘플 요청
CHECK_SAMPLES = {
    "1  특허·실용 공개·등록공보": ("search", {"ipcNumber": "A61K47/68", "numOfRows": 1}),
    "34 대표 출원인": ("rpst_applicant", {"applicationNumber": "1020147035214"}),
    "27 특허 패밀리": ("family", {"applicationNumber": "1020120012361"}),
    "25 피인용문헌": ("citing", {"standardCitationApplicationNumber": "1020080063908"}),
    "48 해외특허": ("foreign_search", {"ipc": "A61K47/68", "collectionValues": "US"}),
}

# --------------------------------------------------------------------------- #
# 주제별 검색식 — 하위 주제 태그(TOPIC_PATTERNS)와 이름을 맞춘다
# --------------------------------------------------------------------------- #
TOPIC_QUERIES: dict[str, list[dict]] = {
    "ADC": [
        {"ipcNumber": "A61K47/68"},                       # 항체-약물 접합체
        {"astrtCont": "항체-약물 접합체"},
    ],
    "AI 신약개발": [
        {"ipcNumber": "G16C20/50"},                       # 분자 설계 (in silico)
        {"ipcNumber": "G16C20/70"},                       # 머신러닝 기반 화학정보
        {"ipcNumber": "G16B40"},                          # 바이오인포매틱스용 머신러닝
        {"astrtCont": "인공지능 신약"},
    ],
    "의료 AI 규제": [                                     # 디지털 헬스케어·의료 AI 기술
        {"ipcNumber": "G16H50/20"},                       # 컴퓨터 보조 진단
        {"ipcNumber": "G16H30/40"},                       # 의료영상 처리
    ],
    "비대면 진료": [
        {"ipcNumber": "G16H40/67"},                       # 원격 의료기기 운용
        {"ipcNumber": "G16H80"},                          # 원격 진료
    ],
    "CDMO": [
        {"astrtCont": "바이오의약품 생산 세포배양"},
    ],
    "바이오시밀러": [
        {"astrtCont": "바이오시밀러"},
    ],
}
# 해외 특허는 IPC 검색식만 사용
FOREIGN_TOPICS = ("ADC", "AI 신약개발", "의료 AI 규제")
# G16C20/70(머신러닝 화학) 등은 의약 외 분야(소재·에너지)도 섞이므로 의약·의료 IPC를 함께 가진 문헌만 남긴다
FOREIGN_IPC_ALLOW = ("A61", "C07", "C12", "G16H", "G16B")


def foreign_relevant(it: dict) -> bool:
    ipcs = [c.strip().replace(" ", "") for c in it.get("ipc", "").split("|")]
    return any(c.startswith(FOREIGN_IPC_ALLOW) for c in ipcs)


class KiprisError(RuntimeError):
    pass


class KiprisBudgetExceeded(KiprisError):
    """--budget 으로 정한 이번 실행의 API 호출 상한에 도달."""


class KiprisKeyError(KiprisError):
    """키 미등록·서비스 미신청 등 — 재시도해도 소용없는 오류."""


# --------------------------------------------------------------------------- #
# 클라이언트
# --------------------------------------------------------------------------- #
def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


class KiprisClient:
    def __init__(self, key: str, interval: float = REQUEST_INTERVAL, use_cache: bool = True, budget: int | None = None):
        self.key = key
        self.interval = interval
        self.use_cache = use_cache
        self.session = requests.Session()
        self._last = 0.0
        self.calls = 0
        self.budget = budget  # 이번 실행에서 허용할 실제 API 호출 수 (캐시 적중은 제외)
        self.cache_hits = 0

    def _cache_path(self, name: str, params: dict) -> Path:
        h = hashlib.sha1(json.dumps([name, params], sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        return CACHE_DIR / name / f"{h}.xml"

    def get(self, name: str, **params) -> ET.Element:
        path, key_param = EP[name]
        cache = self._cache_path(name, params)
        if self.use_cache and cache.exists():
            self.cache_hits += 1
            return self._parse(cache.read_text(encoding="utf-8"), name)

        if self.budget is not None and self.calls >= self.budget:
            raise KiprisBudgetExceeded(f"API 호출 상한 {self.budget}회 도달")
        wait = self.interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                resp = self.session.get(BASE + path, params={**params, key_param: self.key}, timeout=60)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise KiprisError(f"{name} 요청 실패: {e}") from e
                time.sleep(2 ** attempt)
        self._last = time.monotonic()
        self.calls += 1
        resp.encoding = "utf-8"
        root = self._parse(resp.text, name)  # 오류 응답은 여기서 예외 → 캐시하지 않음
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")
        return root

    @staticmethod
    def _parse(text: str, name: str) -> ET.Element:
        try:
            root = _strip_ns(ET.fromstring(text.encode("utf-8")))
        except ET.ParseError as e:
            raise KiprisError(f"{name}: XML 파싱 실패 {text[:200]!r}") from e
        code = (root.findtext(".//resultCode") or "").strip()
        msg = (root.findtext(".//resultMsg") or "").strip()
        success = (root.findtext(".//successYN") or "Y").strip()
        if success == "N" or (code and code not in ("00", "0", "000")):
            if re.search(r"not\s*regist|key", msg, re.I):
                raise KiprisKeyError(f"{name}: {msg} (code {code})")
            raise KiprisError(f"{name}: {msg} (code {code})")
        return root


def items_of(root: ET.Element, tag: str) -> list[dict]:
    """<tag> 요소들을 {자식태그: 텍스트} dict 목록으로."""
    return [{c.tag: (c.text or "").strip() for c in el} for el in root.iter(tag)]


# --------------------------------------------------------------------------- #
# 국내 특허
# --------------------------------------------------------------------------- #
def ipc_variants(ipc: str) -> list[str]:
    """'A61K47/68' ↔ 'A61K 47/68' — 결과가 0건이면 다른 표기로 재시도."""
    compact = ipc.replace(" ", "")
    spaced = re.sub(r"^([A-H]\d{2}[A-Z])(\d)", r"\1 \2", compact)
    return [compact] + ([spaced] if spaced != compact else [])


def search_domestic(client: KiprisClient, query: dict, since: str, until: str, max_items: int) -> list[dict]:
    variants = [query]
    if "ipcNumber" in query:
        variants = [{**query, "ipcNumber": v} for v in ipc_variants(query["ipcNumber"])]
    for q in variants:
        out, page = [], 1
        while len(out) < max_items:
            rows = min(500, max_items - len(out))
            root = client.get(
                "search", **q, applicationDate=f"{since}~{until}", patent="true", utility="true",
                numOfRows=rows, pageNo=page, sortSpec="AD", descSort="true",
            )
            got = items_of(root, "item")
            out += got
            total = int(root.findtext(".//totalCount") or 0)
            if not got or page * rows >= total:
                break
            page += 1
        if out:
            return out[:max_items]
    return []


def first_claim(client: KiprisClient, app_no: str) -> str:
    root = client.get("claim", applicationNumber=app_no)
    claims = [c.text.strip() for c in root.iter("claim") if c.text and c.text.strip()]
    claims = [c for c in claims if "삭제" not in c[:10]]
    return claims[0] if claims else ""


def enrich(client: KiprisClient, app_no: str, what: set[str]) -> dict:
    meta: dict = {}
    if "applicant" in what:
        try:
            info = items_of(client.get("rpst_applicant", applicationNumber=app_no), "rpstApplicantV2Info")
            if info:
                meta["rpst_applicant"] = info[0].get("rpstApplicantName", "")
                meta["rpst_applicant_no"] = info[0].get("rpstApplicantNumber", "")
        except (KiprisKeyError, KiprisBudgetExceeded):
            raise
        except KiprisError as e:
            log.debug("대표출원인 실패 %s: %s", app_no, e)
    if "family" in what:
        try:
            root = client.get("family", applicationNumber=app_no)
            fam = items_of(root, "item") or items_of(root, "Items")
            countries = sorted({f.get("applicationCountryCode", "") for f in fam} - {"", "KR"})
            meta["family_countries"] = countries
            meta["family_size"] = len(fam)
        except (KiprisKeyError, KiprisBudgetExceeded):
            raise
        except KiprisError as e:
            log.debug("패밀리 실패 %s: %s", app_no, e)
    if "citing" in what:
        try:
            cites = items_of(client.get("citing", standardCitationApplicationNumber=app_no), "citingInfo")
            meta["cited_by"] = len({c.get("ApplicationNumber", "") for c in cites} - {""})
        except (KiprisKeyError, KiprisBudgetExceeded):
            raise
        except KiprisError as e:
            log.debug("피인용 실패 %s: %s", app_no, e)
    return meta


def _date(s: str) -> str | None:
    s = re.sub(r"\D", "", s or "")
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}T00:00:00" if len(s) >= 8 else None


def domestic_doc(it: dict, query_name: str, claim: str, extra: dict) -> dict:
    app_no = it.get("applicationNumber", "")
    title = it.get("inventionTitle", "").strip()
    applicant = it.get("applicantName", "").replace("|", ", ")
    ipc = it.get("ipcNumber", "").replace("|", ", ")
    abstract = re.sub(r"\s+", " ", it.get("astrtCont", "")).strip()
    lines = [
        f"발명의 명칭: {title}",
        f"출원인: {applicant}",
        f"출원일: {it.get('applicationDate', '')}  공개일: {it.get('openDate', '')}  등록상태: {it.get('registerStatus', '')}",
        f"IPC: {ipc}",
        f"요약: {abstract}",
    ]
    if claim:
        lines.append(f"대표 청구항: {claim}")
    if extra.get("family_countries"):
        lines.append(f"해외 패밀리 출원국: {', '.join(extra['family_countries'])}")
    body = "\n".join(lines)
    topics = [t for t, p in TOPIC_PATTERNS.items() if re.search(p, f"{title} {abstract}")]
    if query_name in TOPIC_PATTERNS and query_name not in topics:
        topics.append(query_name)
    return {
        "doc_id": f"kipris:KR:{app_no}",
        "source_type": "patent",
        "evidence_role": "leading",
        "title": f"[특허] {title}",
        "body": body,
        "url": KR_DOI.format(app_no),
        # 공개일 = 근거가 공개된 시점. 기술 시점 분석에는 meta.application_date 사용
        "published_at": _date(it.get("openDate")) or _date(it.get("registerDate")) or _date(it.get("applicationDate")),
        "publisher": applicant,
        "collect_keywords": query_name,
        "topics": "; ".join(topics),
        "is_outlook": 0,
        "char_len": len(body),
        "meta": json.dumps(
            {
                "country": "KR",
                "application_number": app_no,
                "application_date": _date(it.get("applicationDate")),
                "open_number": it.get("openNumber", ""),
                "register_number": it.get("registerNumber", ""),
                "register_status": it.get("registerStatus", ""),
                "ipc": it.get("ipcNumber", "").split("|") if it.get("ipcNumber") else [],
                **extra,
            },
            ensure_ascii=False,
        ),
    }


# --------------------------------------------------------------------------- #
# 해외 특허
# --------------------------------------------------------------------------- #
def search_foreign(client: KiprisClient, ipc: str, country: str, since: str, until: str, max_items: int) -> list[dict]:
    for v in ipc_variants(ipc):
        out, page = [], 1
        while len(out) < max_items:
            root = client.get(
                "foreign_search", ipc=v, collectionValues=country, applicationdate=f"{since}~{until}",
                currentPage=page, sortField="AD", sortState="true",
            )
            got = items_of(root, "searchResult")
            out += got
            total = int(root.findtext(".//totalSearchCount") or 0)
            if not got or len(out) >= total:
                break
            page += 1
        if out:
            return out[:max_items]
    return []


def foreign_doc(client: KiprisClient, it: dict, country: str, query_name: str, with_abstract: bool = False) -> dict:
    ltrt = it.get("ltrtno", "")
    abstract = ""
    try:
        if not with_abstract:
            raise KiprisError("초록 조회 생략")
        root = client.get("foreign_abstract", literatureNumber=ltrt, countryCode=country)
        abstract = re.sub(r"\s+", " ", root.findtext(".//astrtCont") or "").strip()
    except (KiprisKeyError, KiprisBudgetExceeded):
        raise
    except KiprisError as e:
        log.debug("해외 초록 실패 %s: %s", ltrt, e)
    title = it.get("inventionName", "").strip()
    # 해외 출원인은 같은 이름이 반복돼 오는 경우가 있어 순서를 유지한 채 중복 제거
    applicant = ", ".join(dict.fromkeys(a.strip() for a in re.split(r"[|,]\s*(?=[A-Z가-힣])", it.get("applicant", "")) if a.strip()))
    body = "\n".join([
        f"Title: {title}",
        f"Applicant: {applicant}",
        f"Country: {country}  Application date: {it.get('applicationDate', '')}  Open date: {it.get('openDate', '')}",
        f"IPC: {it.get('ipc', '').replace('|', ', ')}",
        f"Abstract: {abstract}",
    ])
    return {
        "doc_id": f"kipris:{country}:{ltrt}",
        "source_type": "patent",
        "evidence_role": "leading",
        "title": f"[{country} 특허] {title}",
        "body": body,
        "url": api_url("foreign_search", ipc=it.get("ipc", "").split("|")[0], collectionValues=country, applicationNo=it.get("applicationNo", "")),
        "published_at": _date(it.get("openDate")) or _date(it.get("registerDate")) or _date(it.get("applicationDate")),
        "publisher": applicant,
        "collect_keywords": query_name,
        "topics": query_name if query_name in TOPIC_PATTERNS else "",
        "is_outlook": 0,
        "char_len": len(body),
        "meta": json.dumps(
            {
                "country": country,
                "literature_number": ltrt,
                "application_number": it.get("applicationNo", ""),
                "application_date": _date(it.get("applicationDate")),
                "family_no": it.get("familyNo", ""),
                "ipc": it.get("ipc", "").split("|") if it.get("ipc") else [],
                "lang": "en",
            },
            ensure_ascii=False,
        ),
    }


# --------------------------------------------------------------------------- #
# 실행
# --------------------------------------------------------------------------- #
def check(client: KiprisClient) -> dict[str, str]:
    """서비스별로 명세 샘플 요청을 한 번씩 보내 활성 여부를 확인한다 (캐시 미사용)."""
    client.use_cache = False
    status = {}
    for label, (name, params) in CHECK_SAMPLES.items():
        try:
            client.get(name, **params)
            status[label] = "OK"
        except KiprisError as e:
            status[label] = f"실패 — {e}"
    return status


def applicant_queries() -> dict[str, list[dict]]:
    from collector.crawl_dart import SEED_COMPANIES  # DART 시드 기업과 같은 목록

    return {
        f"출원인:{name}": [{"applicant": name}]
        for names in SEED_COMPANIES.values()
        for name in names
    }


def run(
    since: str = DEFAULT_SINCE,
    until: str | None = None,
    topics: list[str] | None = None,
    by_applicant: bool = False,
    foreign: list[str] | None = None,
    max_per_query: int = 2000,
    enrich_what: set[str] | None = None,
    with_claims: bool = False,
    foreign_abstracts: bool = False,
    foreign_max: int = 30,
    budget: int | None = 100,
    dry_run: bool = False,
    db_path: Path = storage.DEFAULT_DB,
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("KIPRIS_API_KEY")
    if not key:
        raise SystemExit("KIPRIS_API_KEY가 없습니다. .env에 추가하세요 (.env.example 참고).")
    client = KiprisClient(key, budget=budget)
    since_ = since.replace("-", "")
    until_ = (until or date.today().isoformat()).replace("-", "")
    enrich_what = enrich_what or set()  # 무료 한도(월 1,000회) 때문에 기본은 보강 호출 없음

    queries = {k: v for k, v in TOPIC_QUERIES.items() if not topics or k in topics}
    if by_applicant:
        queries |= applicant_queries()

    existing = storage.existing_ids(ORIGIN, db_path)
    docs: dict[str, dict] = {}
    dirty: set[str] = set()
    per_query: dict[str, int] = {}

    def flush() -> None:
        """검색식 하나가 끝날 때마다 저장 — 한도 초과·오류로 중단돼도 그때까지 수집분은 남는다."""
        if dirty and not dry_run:
            storage.save_documents(pd.DataFrame([docs[d] for d in dirty]), ORIGIN, db_path)
        dirty.clear()

    try:
        for qname, qlist in queries.items():
            n = 0
            for q in qlist:
                for it in search_domestic(client, q, since_, until_, max_per_query):
                    app_no = it.get("applicationNumber", "")
                    doc_id = f"kipris:KR:{app_no}"
                    n += 1
                    if not app_no or dry_run:
                        continue
                    if doc_id in docs:  # 여러 검색식에 걸린 특허 → 수집 키워드만 추가
                        kws = set(docs[doc_id]["collect_keywords"].split("; ")) | {qname}
                        docs[doc_id]["collect_keywords"] = "; ".join(sorted(kws))
                        dirty.add(doc_id)
                        continue
                    if doc_id in existing:
                        continue
                    claim = first_claim(client, app_no) if with_claims else ""
                    extra = enrich(client, app_no, enrich_what)
                    docs[doc_id] = domestic_doc(it, qname, claim, extra)
                    dirty.add(doc_id)
            per_query[qname] = n
            flush()
            log.info("%-24s 검색 %4d건 (누적 신규 %d건, API %d회)", qname, n, len(docs), client.calls)

        for country in foreign or []:
            for tname in FOREIGN_TOPICS:
                for q in TOPIC_QUERIES[tname]:
                    if "ipcNumber" not in q:
                        continue
                    hits = search_foreign(client, q["ipcNumber"], country, since_, until_, foreign_max)
                    relevant = [it for it in hits if foreign_relevant(it)]
                    per_query[f"{country}:{tname}:{q['ipcNumber']}"] = len(relevant)
                    if dry_run:
                        continue
                    for it in relevant:
                        doc_id = f"kipris:{country}:{it.get('ltrtno', '')}"
                        if doc_id in docs or doc_id in existing or not it.get("ltrtno"):
                            continue
                        docs[doc_id] = foreign_doc(client, it, country, tname, foreign_abstracts)
                        dirty.add(doc_id)
                    flush()
                    log.info("해외 %s %-28s 검색 %3d건 → 의약·의료 %3d건 (API %d회)",
                             country, f"{tname}:{q['ipcNumber']}", len(hits), len(relevant), client.calls)
    except KiprisKeyError as e:
        log.error("%s → KIPRIS Plus 마이페이지에서 서비스 상태를 확인하세요. 지금까지 수집한 %d건은 저장됐습니다.", e, len(docs))
    except KiprisError as e:
        log.error("%s — 지금까지 수집한 %d건은 저장됐습니다. 다시 실행하면 캐시와 기존 문서를 건너뛰고 이어서 수집합니다.", e, len(docs))
    finally:
        flush()

    df = pd.DataFrame(list(docs.values()))
    stats = {
        "queries": per_query,
        "saved": 0 if dry_run else len(df),
        "by_country": df["meta"].map(lambda m: json.loads(m)["country"]).value_counts().to_dict() if len(df) else {},
        "topics": df["topics"].str.split("; ").explode().loc[lambda s: s.fillna("") != ""].value_counts().to_dict() if len(df) else {},
        "api_calls": client.calls,
        "cache_hits": client.cache_hits,
        "generated_at": storage.now_iso(),
    }
    if not dry_run:
        (db_path.parent / "crawl_kipris_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="KIPRIS Plus OpenAPI 특허 수집 → store/docs.db")
    ap.add_argument("--check", action="store_true", help="서비스별 키 활성 상태만 확인")
    ap.add_argument("--since", default=DEFAULT_SINCE, help="출원일 시작 YYYY-MM-DD")
    ap.add_argument("--until", help="출원일 종료 YYYY-MM-DD (기본: 오늘)")
    ap.add_argument("--topics", nargs="+", choices=list(TOPIC_QUERIES), help="주제 제한")
    ap.add_argument("--by-applicant", action="store_true", help="DART 시드 기업을 출원인으로 검색")
    ap.add_argument("--foreign", nargs="+", metavar="CC", help="해외 특허 국가코드 (US EP WO JP …)")
    ap.add_argument("--foreign-max", type=int, default=30, help="해외 검색식당 최대 건수 (30건=1회 호출)")
    ap.add_argument("--max-per-query", type=int, default=2000, help="검색식당 최대 건수 (국내는 1회 호출에 500건)")
    ap.add_argument("--budget", type=int, default=100, help="이번 실행의 API 호출 상한 (캐시 적중 제외, 0=캐시만 사용)")
    ap.add_argument("--enrich", nargs="+", default=["none"],
                    choices=["applicant", "family", "citing", "none"], help="특허별 보강 정보")
    ap.add_argument("--claims", action="store_true", help="특허마다 대표 청구항 조회 (+1회/건)")
    ap.add_argument("--foreign-abstracts", action="store_true", help="해외 특허마다 영문 초록 조회 (+1회/건)")
    ap.add_argument("--dry-run", action="store_true", help="검색 건수만 확인 (상세 조회·DB 저장 없음)")
    ap.add_argument("--db", type=Path, default=storage.DEFAULT_DB)
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.check:
        load_dotenv(PROJECT_ROOT / ".env")
        key = os.getenv("KIPRIS_API_KEY")
        if not key:
            raise SystemExit("KIPRIS_API_KEY가 없습니다.")
        for label, st in check(KiprisClient(key)).items():
            print(f"{label:<24} {st}")
        return

    enrich_what = set() if "none" in args.enrich else set(args.enrich)
    stats = run(
        args.since, args.until, args.topics, args.by_applicant, args.foreign, args.max_per_query,
        enrich_what, args.claims, args.foreign_abstracts, args.foreign_max, args.budget, args.dry_run, args.db,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""① 수집 계층 — 정부 보도자료 (정책 원문) 수집: 대한민국 정책브리핑(korea.kr).

대표 질문 중 비대면 진료 제도(Q02)·식약처 의료 AI 허가 가이드라인(Q13)·생물보안법 대응(Q14)은
뉴스만으로는 '제도가 실제로 무엇을 정했는가'를 확인할 수 없어 정책 원문(source_type=policy)이 필요하다.

정책브리핑은 각 부처 보도자료를 전재하는 공식 포털로, robots.txt가 `User-Agent: *`에 전체 허용이다.
  - 목록:  /briefing/pressReleaseList.do?repCode={부처}&srchWord={검색어}  (20건/쪽)
  - 상세:  /briefing/pressReleaseView.do?newsId=…   — 제목·날짜·부처·첨부 목록 (본문은 대부분 첨부에만 있음)
  - 첨부:  /common/download.do?fileId=…            — 보도자료 본문 HWPX(XML zip) 우선, 없으면 PDF

준수 사항
  - 1초 간격 순차 요청, 목록·상세·첨부는 data/policy/raw/ 에 캐시해 재실행 시 다시 요청하지 않는다.
  - 보도자료 본문(첫 첨부)만 받고 별첨·붙임(법령 전문·서식 등)은 받지 않는다.
  - 보도자료 하단 '담당 부서/책임자/담당자' 표의 이름·전화번호·이메일은 개인정보이므로 저장하지 않는다.
  - 문서마다 정책브리핑 상세 URL과 수집 일시를 남긴다.

실행:
    python -m collector.crawl_policy                 # 기본 검색식 전체
    python -m collector.crawl_policy --max-pages 2   # 시험 실행
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
from bs4 import BeautifulSoup
from lxml import etree

from collector import storage
from collector.load_provided import OUTLOOK_PATTERN, TOPIC_PATTERNS

log = logging.getLogger("collector.crawl_policy")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "policy" / "raw"
BASE = "https://www.korea.kr"
ORIGIN = "korea_kr"
DEFAULT_SINCE = "2021-10-16"
REQUEST_INTERVAL = 1.0
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mini-proj3-research-crawler; educational use)"}
MIN_BODY_CHARS = 400
MAX_PAGES = 10                     # 검색식당 최대 목록 쪽수 (20건/쪽)

MINISTRIES = {
    "A00012": "보건복지부",
    "A00027": "식품의약품안전처",
    "A00033": "과학기술정보통신부",
    "A00015": "산업통상부",
}
# 부처별 검색어 (정책브리핑 검색은 첨부 본문까지 찾으므로 제목·요약으로 다시 거른다)
QUERIES = {
    "A00012": ["비대면진료", "비대면 진료", "디지털헬스", "디지털 헬스케어", "바이오헬스", "제약바이오", "인공지능", "의료데이터", "첨단재생"],
    "A00027": ["인공지능", "디지털의료제품", "디지털의료기기", "바이오의약품", "바이오시밀러", "항체", "의료기기 허가", "신약"],
    "A00033": ["바이오", "신약", "디지털 헬스", "의료 인공지능"],
    "A00015": ["바이오", "생물보안", "의약품", "CDMO"],
}
# 제목 핵심어 (AI·인공지능 단독은 의료 맥락이 함께 있을 때만 인정)
TITLE_PATTERN = re.compile(
    r"바이오|제약|의약품|신약|항체|ADC|CDMO|위탁\s?생산|위탁\s?개발|바이오시밀러|동등생물|임상\s?시험|의료\s?기기|"
    r"디지털\s?의료|디지털\s?헬스|헬스\s?케어|비대면\s?진료|원격\s?(의료|협진)|의료\s?(AI|인공지능)|의료\s?데이터|"
    r"보건의료\s?데이터|(신|혁신)\s?의료\s?기술|첨단\s?재생|세포\s?치료|유전자\s?치료|백신|생물\s?보안|보건\s?산업|의료\s?제품",
    re.I,
)
AI_PATTERN = re.compile(r"AI|인공지능", re.I)
MEDICAL_CONTEXT = re.compile(r"의료|헬스|신약|진단|병원|의약|임상|보건|영상|판독")
TITLE_EXCLUDE = re.compile(
    r"인사|채용|공모전|포스터|취임|이임|신년사|예산|식품|농|축산|수산|화장품|마약|담배|금연|주류|위생|급식|"
    r"어린이|장애인|연금|돌봄|보육|아동|노인|복지용구|사회서비스|ODA|건강보험|수가|상대가치"
)
LEAD_MIN_HITS = 3                   # 제목에 핵심어가 없을 때 요약에 핵심어가 이만큼 있어야 인정

PHONE = re.compile(r"\(?0\d{1,2}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
CONTACT_LINE = re.compile(r"담당\s?부서|책임자|담당자|사무관|주무관|연구관|연구사|과\s?장|팀\s?장|<\s?총괄|<\s?공동")


class Fetcher:
    def __init__(self, interval: float = REQUEST_INTERVAL):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.interval = interval
        self._last = 0.0
        self.requests = 0
        self.cache_hits = 0

    def get_bytes(self, url: str, cache: Path) -> bytes:
        if cache.exists():
            self.cache_hits += 1
            return cache.read_bytes()
        wait = self.interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=60)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                log.warning("재시도 %s: %s", url, e)
                time.sleep(3 * (attempt + 1))
        self._last = time.monotonic()
        self.requests += 1
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(resp.content)
        return resp.content

    def get_html(self, url: str, cache: Path) -> BeautifulSoup:
        return BeautifulSoup(self.get_bytes(url, cache).decode("utf-8", "replace"), "lxml")


# --------------------------------------------------------------------------- #
# 목록 · 상세 · 첨부
# --------------------------------------------------------------------------- #
def list_page(f: Fetcher, rep: str, word: str, page: int, since: str, until: str) -> tuple[list[dict], int]:
    q = {"pageIndex": page, "repCodeType": "", "repCode": rep, "startDate": since, "endDate": until, "srchWord": word}
    safe = re.sub(r"\W", "_", word)
    soup = f.get_html(f"{BASE}/briefing/pressReleaseList.do?{urlencode(q)}",
                      RAW_DIR / "list" / f"{rep}_{safe}_{since}_{until}_{page}.html")
    total = soup.select_one("div.sch_brief strong")
    total = int(re.sub(r"\D", "", total.get_text()) or 0) if total else 0
    items = []
    for li in soup.select("div.list_type li"):
        a = li.select_one("a[href*='pressReleaseView.do']")
        if not a:
            continue
        nid = re.search(r"newsId=(\d+)", a["href"]).group(1)
        src = [s.get_text(strip=True) for s in li.select("span.source span")]
        lead = li.select_one("span.lead")
        items.append({
            "news_id": nid,
            "title": li.select_one("strong").get_text(" ", strip=True),
            "lead": lead.get_text(" ", strip=True) if lead else "",
            "date": src[0] if src else "",
            "ministry": src[1] if len(src) > 1 else "",
        })
    return items, total


def relevant(item: dict) -> bool:
    title = item["title"]
    if TITLE_EXCLUDE.search(title) and not re.search(r"바이오|의약품|의료\s?기기|디지털\s?(의료|헬스)", title):
        return False
    if TITLE_PATTERN.search(title):
        return True
    if AI_PATTERN.search(title) and MEDICAL_CONTEXT.search(title):
        return True
    return len(TITLE_PATTERN.findall(item["lead"])) >= LEAD_MIN_HITS


def parse_view(soup: BeautifulSoup) -> dict:
    title = soup.select_one("div.view_title")
    info = soup.select_one("div.info")
    files = []
    for p in soup.select("div.filedown p"):
        a = p.select_one("a[href*='/common/download.do']")
        if not a:
            continue
        fid = re.search(r"fileId=(\d+)", a["href"]).group(1)
        tbl = re.search(r"tblKey=(\w+)", a["href"])
        files.append({"file_id": fid, "tbl": tbl.group(1) if tbl else "GMN", "name": a.get_text(" ", strip=True)})
    cont = soup.select_one("div.view_cont")
    return {
        "title": title.get_text(" ", strip=True) if title else "",
        "info": info.get_text(" ", strip=True) if info else "",
        "inline": cont.get_text("\n", strip=True) if cont else "",
        "files": files,
    }


def pick_main_file(files: list[dict]) -> dict | None:
    """보도자료 본문 파일 1개: 별첨·붙임·참고가 아닌 첫 HWPX, 없으면 첫 PDF."""
    main = [x for x in files if not re.match(r"^\s*[\(\[]?\s*(별첨|붙임|참고|첨부|별지)", x["name"])] or files
    for ext in (".hwpx", ".pdf"):
        for x in main:
            if x["name"].lower().endswith(ext):
                return x
    return None


def hwpx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        sections = sorted(n for n in z.namelist() if re.match(r"Contents/section\d+\.xml$", n))
        paras = []
        for n in sections:
            root = etree.fromstring(z.read(n))
            for p in root.iter("{*}p"):
                # 표를 품은 문단은 안쪽 셀 문단이 따로 다시 나오므로 건너뛴다 (표 내용 이중 추출 방지)
                if next(p.iter("{*}tbl"), None) is not None:
                    continue
                t = "".join(x.text or "" for x in p.iter("{*}t"))
                if t.strip():
                    paras.append(t.strip())
    return "\n".join(paras)


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    return "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(data)).pages)


CONTACT_HEAD = re.compile(r"^(과\s?장|팀\s?장|부\s?장|단\s?장|책임자|담당자|담당\s?부서|사무관|주무관|연구관|연구사)$")
HEAD_MARK = re.compile(r"보\s?도\s?(자\s?료|일\s?시|참\s?고)|배\s?포|전\s?화|과\s?장|담\s?당|사무관|주무관|연구관")
HEAD_BLOCK_LINES = 25
BARE_NAME = re.compile(r"[가-힣]{2,4}")                                            # 이름만 있는 줄
ROLE_LINE = re.compile(r"(과|팀|부|단|국|실)\s*장|서기관|사무관|주무관|연구관|연구사|책임자|담당자|행정관|기술서기관|보건연구관")
ORG_LINE = re.compile(r"[가-힣·\d]{2,30}(과|관|국|팀|단|실|사무국|본부|위원회)")  # 부서명만 있는 줄
HEADER_LINE = re.compile(r"^(보도\s?자료|보도\s?시점.*|배\s?포.*|\d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.?\s?\(.\).*|\d{1,2}:\d{2})$")


def scrub(text: str) -> tuple[str, int]:
    """보도자료 하단 담당자 표(부서·직위·이름)를 통째로 잘라내고, 전화·이메일이 있는 줄과 머리말을 지운다.
    반환: (본문, 제거한 줄 수)"""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    n0 = len(lines)
    # 담당자 표: 문서 뒤쪽 절반에서 처음 나오는 '과 장/담당자…' 칸 → 바로 앞의 짧은 부서명 줄까지 포함해 끝까지 제거
    half = len(lines) // 2
    cut = next((i for i in range(half, len(lines)) if CONTACT_HEAD.match(lines[i])), None)
    if cut is not None:
        while cut > half and len(lines[cut - 1]) <= 15 and not re.search(r"[.다)」]$", lines[cut - 1]):
            cut -= 1
        lines = lines[:cut]
    # 옛 서식 머리 블록: 제목 앞에 '보도 일시·부서·과 장·이름·전화'가 짧은 줄로 나열됨 → 제목(긴 줄) 전까지 짧은 줄 제거
    top = lines[:HEAD_BLOCK_LINES]
    if any(HEAD_MARK.search(x) for x in top):
        start = next((i for i, x in enumerate(top)
                      if len(x) >= 13 and len(re.findall(r"[가-힣]", x)) >= 8 and not HEAD_MARK.search(x)), len(top))
        lines = lines[start:]
        # 제목으로 잘못 잡힌 긴 부서명 뒤에 남은 '과 장·이름' 줄도 머리 블록 안에서 지운다
        lines = [x for i, x in enumerate(lines)
                 if not (i < HEAD_BLOCK_LINES and (BARE_NAME.fullmatch(x) or ORG_LINE.fullmatch(x) or CONTACT_HEAD.match(x)))]
    # 문서 어디서든: 직위만 있는 줄(서기관·사무관·과 장 …)과 바로 뒤 2줄 안의 이름 줄을 지운다 (짧은 문서의 담당자 표)
    drop = set()
    for i, x in enumerate(lines):
        if ROLE_LINE.fullmatch(x):
            drop.add(i)
            drop.update(j for j in range(i + 1, min(i + 3, len(lines))) if BARE_NAME.fullmatch(lines[j]))
    lines = [x for i, x in enumerate(lines) if i not in drop]
    out = [x for x in lines if not (PHONE.search(x) or EMAIL.search(x) or HEADER_LINE.match(x)
                                    or (len(x) < 40 and CONTACT_LINE.search(x)))]
    removed = n0 - len(out)
    body = "\n".join(out)
    body = re.sub(r"[ \t ]+", " ", body)
    return body, removed


# --------------------------------------------------------------------------- #
# 수집
# --------------------------------------------------------------------------- #
def to_document(item: dict, view: dict, body: str, body_from: str, matched: list[str]) -> dict:
    text = f"{item['title']}\n{body}"
    topics = [t for t, p in TOPIC_PATTERNS.items() if re.search(p, text, re.I)]
    is_outlook = int(bool(re.search(OUTLOOK_PATTERN, text)))
    return {
        "doc_id": f"korea_kr:{item['news_id']}",
        "source_type": "policy",
        "evidence_role": "outlook" if is_outlook and re.search(r"전략|로드맵|계획|추진방안", item["title"]) else "current",
        "title": item["title"],
        "body": body,
        "url": f"{BASE}/briefing/pressReleaseView.do?newsId={item['news_id']}",
        "published_at": pd.to_datetime(item["date"].replace(".", "-")).isoformat(),
        "publisher": item["ministry"],
        "collect_keywords": "; ".join(sorted(set(matched))),
        "topics": "; ".join(topics),
        "is_outlook": is_outlook,
        "keyword_hits": len(TITLE_PATTERN.findall(text)),
        "char_len": len(body),
        "source_files": body_from,
        "meta": json.dumps({"site": "정책브리핑(korea.kr)", "ministry": item["ministry"], "lead": item["lead"],
                            "attachments": [x["name"] for x in view["files"]]}, ensure_ascii=False),
        "loaded_at": storage.now_iso(),
    }


def run(since: str = DEFAULT_SINCE, until: str | None = None, max_pages: int = MAX_PAGES,
        db_path: Path = storage.DEFAULT_DB) -> dict:
    until = until or pd.Timestamp.today().strftime("%Y-%m-%d")
    f = Fetcher()
    found: dict[str, dict] = {}
    matched: dict[str, list[str]] = {}
    stats = {"queries": {}, "list_items": 0}
    for rep, words in QUERIES.items():
        for w in words:
            n_rel = 0
            for page in range(1, max_pages + 1):
                items, total = list_page(f, rep, w, page, since, until)
                stats["list_items"] += len(items)
                for it in items:
                    if relevant(it):
                        n_rel += 1
                        found.setdefault(it["news_id"], it)
                        matched.setdefault(it["news_id"], []).append(w)
                if not items or page * 20 >= total:
                    break
            stats["queries"][f"{MINISTRIES[rep]}:{w}"] = {"total": total, "relevant": n_rel}
            log.info("%s '%s' 검색 %d건 → 관련 %d건 (누적 고유 %d)", MINISTRIES[rep], w, total, n_rel, len(found))

    docs, rejected, removed_lines = [], [], 0
    for i, (nid, it) in enumerate(sorted(found.items()), 1):
        view = parse_view(f.get_html(f"{BASE}/briefing/pressReleaseView.do?newsId={nid}", RAW_DIR / "view" / f"{nid}.html"))
        main = pick_main_file(view["files"])
        body, body_from = "", "inline"
        if main:
            ext = Path(main["name"]).suffix.lower()
            try:
                data = f.get_bytes(f"{BASE}/common/download.do?fileId={main['file_id']}&tblKey={main['tbl']}",
                                   RAW_DIR / "file" / f"{main['file_id']}{ext}")
                body = hwpx_text(data) if ext == ".hwpx" else pdf_text(data)
                body_from = ext.lstrip(".")
            except Exception as e:  # 손상 파일 등은 요약으로 대체
                log.warning("첨부 파싱 실패 %s: %s", nid, e)
        if len(body) < MIN_BODY_CHARS and len(view["inline"]) > len(body):
            body, body_from = view["inline"], "inline"
        body, n = scrub(body)
        removed_lines += n
        if len(body) < MIN_BODY_CHARS:
            rejected.append({"doc_id": f"korea_kr:{nid}", "title": it["title"],
                             "url": f"{BASE}/briefing/pressReleaseView.do?newsId={nid}",
                             "published_at": it["date"], "drop_reason": "short_or_no_body", "char_len": len(body),
                             "loaded_at": storage.now_iso()})
            continue
        docs.append(to_document(it, view, body, body_from, matched[nid]))
        if i % 50 == 0:
            log.info("상세 %d/%d (요청 %d, 캐시 %d)", i, len(found), f.requests, f.cache_hits)

    frame = pd.DataFrame(docs)
    storage.save_documents(frame, ORIGIN, db_path, rejected=pd.DataFrame(rejected) if rejected else None)
    stats.update({
        "unique_relevant": len(found),
        "saved": len(docs),
        "rejected": len(rejected),
        "by_ministry": frame["publisher"].value_counts().to_dict() if len(frame) else {},
        "body_from": frame["source_files"].value_counts().to_dict() if len(frame) else {},
        "contact_lines_removed": removed_lines,
        "requests": f.requests,
        "cache_hits": f.cache_hits,
        "period": [since, until],
        "generated_at": storage.now_iso(),
    })
    (Path(db_path).parent / "crawl_policy_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="정책브리핑 보도자료 수집 → documents(policy)")
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--until")
    ap.add_argument("--max-pages", type=int, default=MAX_PAGES)
    ap.add_argument("--db", type=Path, default=storage.DEFAULT_DB)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.since, args.until, args.max_pages, args.db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

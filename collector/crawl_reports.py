"""① 수집 계층 — 산업보고서 크롤링 (BioIN 생명공학정책연구센터 포털).

수집 게시판 (https://www.bioin.or.kr)
  watch     BioINdustry  — 산업·시장 현황 및 전망 분석 보고서
  report    BioINpro     — 전문가 기술·산업 보고서
  global    BioINglobal  — 해외 동향 보고서
  industry  산업 동향    — 유관기관 브리프 모음 (KOTRA, 한국보건산업진흥원, 한국바이오협회, 산업연구원 등)

준수 사항
  - robots.txt: bioin.or.kr 은 /innorix/, /InnoAP/, /sn3hcv/, /sso/ 만 금지 → 게시판 목록·본문 HTML만 요청,
    첨부파일(PDF) 다운로드는 하지 않음. (data/compliance/www.bioin.or.kr_robots.txt)
  - 요청 간격 REQUEST_INTERVAL초, 순차 요청, 로그인 불필요한 공개 게시물만
  - 문서마다 출처 URL(BioIN 게시물 주소)과 원 발행기관·원문 링크(meta), 수집 일시(loaded_at) 기록
  - 사이트가 생성한 'AI 요약'은 원문이 아니므로 제외하고 게시물 본문(div.contents)만 저장

한국보건산업진흥원(khidi.or.kr) 보고서는 본문이 PDF로만 제공되고 robots.txt가 /fileDownload 를 금지하므로
직접 수집하지 않는다. 진흥원의 '글로벌 바이오헬스산업 동향' 등은 BioIN 산업 동향 게시판에 본문으로 재게시된 것을 받는다.

실행 예:
    python -m collector.crawl_reports                      # 4개 게시판, 2021-10-16 이후
    python -m collector.crawl_reports --boards watch global --max-per-board 20
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from collector import storage
from collector.load_provided import OUTLOOK_PATTERN, TOPIC_PATTERNS

log = logging.getLogger("collector.crawl_reports")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "reports" / "raw"
BASE = "https://www.bioin.or.kr"
ORIGIN = "bioin"
DEFAULT_SINCE = "2021-10-16"
REQUEST_INTERVAL = 1.0
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mini-proj3-research-crawler; educational use)"}

BOARDS = {
    "watch": "BioINdustry",
    "report": "BioINpro",
    "global": "BioINglobal",
    "industry": "산업 동향",
}

# 산업 동향 게시판 분류 중 바이오·헬스케어 관련 (농림·식품·축산·환경·에너지 제외)
INDUSTRY_CATEGORIES = {
    "바이오의약", "의약기술", "바이오의료기기", "의료기기기술", "보건의료학", "바이오서비스",
    "의료서비스기술", "종합", "생명과학", "바이오융합기술", "바이오기반기술", "기타",
}
# 분류가 넓은 경우에는 제목에 바이오·헬스 도메인어가 있어야 수집
BROAD_CATEGORIES = {"종합", "생명과학", "바이오융합기술", "바이오기반기술", "기타"}
DOMAIN_PATTERN = r"바이오|제약|의약|헬스|의료|신약|항체|ADC|CDMO|CMO|세포|유전자|임상|진단|백신|치료제|디지털|AI|인공지능|병원|규제|허가"
EXCLUDE_PATTERN = r"농업|농식품|축산|수산|식품|작물|종자|산림|해양|비료|사료|에너지|탄소|기후"
MIN_BODY_CHARS = 300
STOP_AFTER_OLD = 5  # 날짜가 수집 기간보다 오래된 게시물이 연속 N건이면 그 게시판 중단


class Fetcher:
    def __init__(self, interval: float = REQUEST_INTERVAL):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.interval = interval
        self._last = 0.0
        self.requests = 0
        self.cache_hits = 0

    def get(self, url: str, cache: Path | None = None) -> str:
        if cache and cache.exists():
            self.cache_hits += 1
            return cache.read_text(encoding="utf-8")
        wait = self.interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == 2:
                    raise
                log.warning("재시도 %s: %s", url, e)
                time.sleep(3 * (attempt + 1))
        self._last = time.monotonic()
        self.requests += 1
        resp.encoding = resp.encoding or "utf-8"
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(resp.text, encoding="utf-8")
        return resp.text


# --------------------------------------------------------------------------- #
# 목록
# --------------------------------------------------------------------------- #
def list_page(f: Fetcher, bid: str, page: int, per_page: int = 50) -> list[dict]:
    html = f.get(f"{BASE}/board.do?bid={bid}&cmd=list&cPage={page}&listCnt={per_page}")
    soup = BeautifulSoup(html, "lxml")
    items: dict[str, dict] = {}
    for a in soup.find_all("a", href=True):
        m = re.match(rf"/board/{bid}/(\d+)", a["href"])
        if not m or m.group(1) in items:
            continue
        li = a.find_parent("li")
        text = li.get_text(" | ", strip=True) if li else a.get_text(" ", strip=True)
        parts = [p for p in text.split(" | ") if p != "이동"]
        d = re.search(r"작성일 \| (\d{4}-\d{2}-\d{2})", text)
        items[m.group(1)] = {
            "num": m.group(1),
            "category": parts[0] if bid == "industry" and parts else "",
            "title": parts[1] if bid == "industry" and len(parts) > 1 else (parts[0] if parts else ""),
            "list_text": text,
            "list_date": d.group(1) if d else None,
        }
    return list(items.values())


def relevant(bid: str, item: dict) -> bool:
    title = item["title"]
    if re.search(EXCLUDE_PATTERN, title) and not re.search(r"의약|헬스|의료|제약", title):
        return False
    if bid == "industry":
        cat = item["category"]
        if cat not in INDUSTRY_CATEGORIES:
            return False
        if cat in BROAD_CATEGORIES:
            return bool(re.search(DOMAIN_PATTERN, title))
        return True
    return bool(re.search(DOMAIN_PATTERN, item["list_text"][:400]))


# --------------------------------------------------------------------------- #
# 본문
# --------------------------------------------------------------------------- #
def _field(view: BeautifulSoup, label: str) -> str:
    for li in view.select("ul.info li, div.viewbox li"):
        t = li.get_text(" ", strip=True)
        if t.startswith(label):
            return t[len(label):].strip()
    return ""


def parse_view(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    view = soup.select_one("article.board_view")
    if view is None:
        return {}
    title = view.select_one("h1.title")
    body_el = view.select_one("div.contents")
    if body_el:
        for t in body_el(["script", "style"]):
            t.decompose()
        lines = [ln.strip() for ln in body_el.get_text("\n").split("\n") if ln.strip()]
        # 글머리 기호(□ ○ ※ 【 등)만 있는 줄은 다음 줄과 합친다
        merged: list[str] = []
        for ln in lines:
            if merged and re.fullmatch(r"[□■○●◦\-※▶◈【】\[\]·∙]+", merged[-1]):
                merged[-1] = f"{merged[-1]} {ln}"
            else:
                merged.append(ln)
        body = "\n".join(merged)
        body = re.sub(r"\n?☞ 자세한 내용은.*$", "", body, flags=re.S).strip()
    else:
        body = ""
    keywords = [k.strip() for k in _field(view, "키워드").split("#") if k.strip()]
    link = view.select_one("div.viewbox a[href^=http]")
    return {
        "title": title.get_text(" ", strip=True) if title else "",
        "registered": _field(view, "등록일")[:10],
        "published": (_field(view, "자료발간일") or _field(view, "발간일"))[:10],
        "source": _field(view, "출처"),
        "category_path": _field(view, "분류"),
        "keywords": keywords,
        "original_url": link["href"] if link else "",
        "body": body,
    }


def normalize_publisher(src: str) -> str:
    """'Kotra', 'KOTRA  해외시장뉴스' → 'KOTRA' 등 발행기관 표기 통일."""
    s = re.sub(r"\s+", " ", src or "").strip()
    if re.match(r"(?i)kotra|대한무역투자진흥공사", s):
        return "KOTRA"
    if s.startswith("한국바이오협회"):
        return "한국바이오협회"
    return s


def to_document(bid: str, item: dict, v: dict) -> dict:
    url = f"{BASE}/board.do?cmd=view&bid={bid}&num={item['num']}"
    date_ = v["published"] or v["registered"] or item.get("list_date") or ""
    text = f"{v['title']}\n{v['body']}"
    topics = [t for t, p in TOPIC_PATTERNS.items() if re.search(p, text)]
    outlook = bool(re.search(OUTLOOK_PATTERN, text))
    publisher = normalize_publisher(v["source"]) or "국가생명공학정책연구센터"
    header = f"[{BOARDS[bid]}] {v['title']}\n발행: {publisher}  발간일: {date_}"
    if v["keywords"]:
        header += f"\n키워드: {', '.join(v['keywords'])}"
    body = f"{header}\n\n{v['body']}"
    return {
        "doc_id": f"bioin:{bid}:{item['num']}",
        "source_type": "report",
        # 시장 규모·성장률 등 전망 서술이 있으면 전망 근거, 아니면 현재 동향 근거
        "evidence_role": "outlook" if outlook else "current",
        "title": f"[보고서] {v['title']}",
        "body": body,
        "url": url,
        "published_at": f"{date_}T00:00:00" if re.match(r"\d{4}-\d{2}-\d{2}", date_) else None,
        "publisher": publisher,
        "collect_keywords": BOARDS[bid],
        "topics": "; ".join(topics),
        "is_outlook": int(outlook),
        "char_len": len(body),
        "meta": json.dumps(
            {
                "site": "bioin.or.kr",
                "board": bid,
                "num": item["num"],
                "category": item.get("category") or "",
                "category_path": v["category_path"],
                "keywords": v["keywords"],
                "original_publisher": v["source"],
                "original_url": v["original_url"],
                "registered": v["registered"],
            },
            ensure_ascii=False,
        ),
    }


# --------------------------------------------------------------------------- #
# 실행
# --------------------------------------------------------------------------- #
def crawl_board(f: Fetcher, bid: str, since: str, until: str, skip: set[str], max_docs: int | None) -> tuple[list[dict], dict]:
    docs, stat = [], {"listed": 0, "relevant": 0, "saved": 0, "short": 0}
    old_streak, page = 0, 1
    while True:
        items = list_page(f, bid, page)
        if not items:
            break
        for it in items:
            stat["listed"] += 1
            if it["list_date"] and it["list_date"] < since:
                old_streak += 1
                if old_streak >= STOP_AFTER_OLD:
                    return docs, stat
                continue
            if not relevant(bid, it):
                continue
            stat["relevant"] += 1
            doc_id = f"bioin:{bid}:{it['num']}"
            if doc_id in skip:
                continue
            html = f.get(f"{BASE}/board.do?cmd=view&bid={bid}&num={it['num']}", CACHE_DIR / f"{bid}_{it['num']}.html")
            v = parse_view(html)
            date_ = v.get("published") or v.get("registered") or ""
            if date_ and date_ < since:
                old_streak += 1
                if old_streak >= STOP_AFTER_OLD:
                    return docs, stat
                continue
            old_streak = 0
            if date_ and date_ > until:
                continue
            if len(v.get("body", "")) < MIN_BODY_CHARS:
                stat["short"] += 1
                continue
            docs.append(to_document(bid, it, v))
            stat["saved"] += 1
            if max_docs and len(docs) >= max_docs:
                return docs, stat
        page += 1
    return docs, stat


def run(
    boards: list[str] | None = None,
    since: str = DEFAULT_SINCE,
    until: str | None = None,
    max_per_board: int | None = None,
    db_path: Path = storage.DEFAULT_DB,
) -> dict:
    until = until or date.today().isoformat()
    f = Fetcher()
    skip = storage.existing_ids(ORIGIN, db_path)
    stats: dict = {"boards": {}}
    total = 0
    for bid in boards or list(BOARDS):
        try:
            docs, st = crawl_board(f, bid, since, until, skip, max_per_board)
        except requests.RequestException as e:
            log.error("%s 게시판 중단: %s", bid, e)
            docs, st = [], {"error": str(e)}
        if docs:
            storage.save_documents(pd.DataFrame(docs), ORIGIN, db_path)  # 게시판마다 저장
        total += len(docs)
        stats["boards"][bid] = st
        log.info("%-9s 목록 %s건 → 관련 %s건 → 저장 %d건 (요청 %d회, 캐시 %d회)",
                 bid, st.get("listed"), st.get("relevant"), len(docs), f.requests, f.cache_hits)
    stats |= {"saved": total, "requests": f.requests, "cache_hits": f.cache_hits, "generated_at": storage.now_iso()}
    (db_path.parent / "crawl_reports_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="BioIN 산업보고서 수집 → store/docs.db")
    ap.add_argument("--boards", nargs="+", choices=list(BOARDS))
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--until")
    ap.add_argument("--max-per-board", type=int)
    ap.add_argument("--db", type=Path, default=storage.DEFAULT_DB)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run(args.boards, args.since, args.until, args.max_per_board, args.db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

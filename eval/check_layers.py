"""계층 분리 검증 — 정적 검사(import·호출) + 실행 검사(실제 앱 구동).

    python -m eval.check_layers            # 정적 + 실행 (실행 검사는 질의 임베딩 2회·답변 생성 1회 API 호출)
    python -m eval.check_layers --static   # 정적 검사만 (API 호출 없음)

계층 규칙 (README §2)
- ① 수집기(collector): 차트·UI 코드 금지.
- ④ 챗봇(app, rag): 문서 임베딩 생성·크롤링·인덱스 쓰기 금지. 수집기·인덱서 코드에 의존하지 않는다.

정적 검사: 모든 .py의 AST를 읽어 함수 안의 지연 import까지 포함해 검사한다.
실행 검사: 새 프로세스에서 collector·indexer import를 막고, 저장소 쓰기 경로(Chroma 쓰기, 문서 임베딩,
docs.db 쓰기 모드 연결)를 가로채 호출 시 기록한다. 그 상태에서 streamlit AppTest로 app/main.py를 실제로 구동해
검색 탭 검색 1회, 대화 탭 질문 1회, 성능 분석 탭 렌더링을 수행한다. 끝나면 store/ 파일의 크기·수정 시각을 대조한다.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

UI_LIBS = {"streamlit", "altair", "matplotlib", "plotly", "seaborn"}
CRAWL_LIBS = {"requests", "bs4", "lxml", "urllib.request", "selenium", "playwright", "httpx"}
# 인덱스·문서 쓰기를 뜻하는 호출 이름 (dict.update·set.add와 겹치는 add/update/delete는 실행 검사에서 잡는다)
WRITE_CALLS = {"add_documents", "add_texts", "from_documents", "from_texts", "upsert", "embed_documents",
               "create_collection", "get_or_create_collection", "delete_collection", "reset", "persist"}
SQL_WRITE = re.compile(r"^\s*(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|REPLACE)\b", re.I)

RULES = {
    "collector": {"forbid_imports": UI_LIBS | {"app"}, "why": "수집기: 차트·UI 코드 금지"},
    "indexer": {"forbid_imports": UI_LIBS | {"app"}, "why": "인덱서: UI 코드 금지"},
    "rag": {"forbid_imports": {"collector", "indexer"} | CRAWL_LIBS | UI_LIBS, "forbid_writes": True,
            "why": "챗봇 체인: 수집기·인덱서 의존·크롤링·UI·인덱스 쓰기 금지"},
    "app": {"forbid_imports": {"collector", "indexer"} | CRAWL_LIBS, "forbid_writes": True,
            "why": "챗봇 UI: 수집기·인덱서 의존·크롤링·인덱스 쓰기 금지"},
}


def _mods(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
        return [node.module]
    return []


def _hit(mod: str, banned: set[str]) -> str | None:
    return next((b for b in banned if mod == b or mod.startswith(b + ".")), None)


def static_check() -> dict:
    report: dict = {}
    for layer, rule in RULES.items():
        files = sorted((ROOT / layer).glob("*.py"))
        violations, imports = [], set()
        for f in files:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
            rel = f.relative_to(ROOT).as_posix()
            for node in ast.walk(tree):
                for m in _mods(node):
                    imports.add(m.split(".")[0])
                    if b := _hit(m, rule["forbid_imports"]):
                        violations.append(f"{rel}:{node.lineno} import {m} (금지: {b})")
                if not rule.get("forbid_writes"):
                    continue
                if isinstance(node, ast.Call):
                    fn = node.func
                    name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
                    if name in WRITE_CALLS:
                        violations.append(f"{rel}:{node.lineno} 쓰기 호출 {name}()")
                    if name == "connect" and isinstance(fn, ast.Attribute) and getattr(fn.value, "id", "") == "sqlite3":
                        arg = node.args[0] if node.args else None
                        src = ast.unparse(arg) if arg is not None else ""
                        if "mode=ro" not in src:
                            violations.append(f"{rel}:{node.lineno} sqlite3.connect 읽기 전용(mode=ro) 아님")
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and SQL_WRITE.match(node.value):
                    violations.append(f"{rel}:{node.lineno} 쓰기 SQL: {node.value.strip()[:40]}")
        report[layer] = {"rule": rule["why"], "files": len(files),
                         "project_imports": sorted(imports & {"collector", "indexer", "rag", "app", "eval"}),
                         "violations": violations}
    return report


# ---------------------------------------------------------------- 실행 검사 (자식 프로세스에서 실행)

RUNTIME = r'''
import importlib.abc, json, sqlite3, sys, time
from pathlib import Path
ROOT = Path(sys.argv[1]); sys.path.insert(0, str(ROOT))
events = {"blocked_imports": [], "writes": []}

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("collector", "indexer"):
            events["blocked_imports"].append(name)
            raise ImportError(f"계층 검사: {name} import 차단")
        return None
sys.meta_path.insert(0, Block())

def trap(owner, attr, label):
    orig = getattr(owner, attr)
    def wrapped(*a, **k):
        events["writes"].append(label)
        raise RuntimeError(f"계층 검사: {label} 차단")
    setattr(owner, attr, wrapped)

from chromadb.api.models.Collection import Collection
for m in ("add", "upsert", "update", "delete", "modify"):
    trap(Collection, m, f"chroma.Collection.{m}")
import chromadb.api.client as cc
for m in ("create_collection", "delete_collection", "reset"):
    if hasattr(cc.Client, m):
        trap(cc.Client, m, f"chroma.Client.{m}")
# langchain_chroma.Chroma()는 내부에서 get_or_create_collection을 부른다.
# 이미 있는 컬렉션이면 '조회'이므로 허용하고, 새로 만들게 되는 경우만 쓰기로 기록한다.
_goc = cc.Client.get_or_create_collection
def get_or_create(self, name, *a, **k):
    names = [c.name if hasattr(c, "name") else c for c in self.list_collections()]
    if name not in names:
        events["writes"].append(f"chroma.Client.get_or_create_collection({name}) — 새 컬렉션 생성")
        raise RuntimeError("계층 검사: 컬렉션 생성 차단")
    events.setdefault("chroma_opened", []).append(name)
    return _goc(self, name, *a, **k)
cc.Client.get_or_create_collection = get_or_create
from langchain_openai import OpenAIEmbeddings
# langchain_openai의 embed_query는 내부에서 embed_documents([질문])을 부른다 → 질의 임베딩 경로만 허용한다.
_eq, _ed = OpenAIEmbeddings.embed_query, OpenAIEmbeddings.embed_documents
_in_query = {"on": False}
def embed_query(self, text, **k):
    _in_query["on"] = True
    try:
        events.setdefault("query_embeddings", []).append(text[:40])
        return _eq(self, text, **k)
    finally:
        _in_query["on"] = False
def embed_documents(self, texts, **k):
    if _in_query["on"] and len(texts) == 1:
        return _ed(self, texts, **k)
    events["writes"].append(f"OpenAIEmbeddings.embed_documents (문서 {len(texts)}개)")
    raise RuntimeError("계층 검사: 문서 임베딩 차단")
OpenAIEmbeddings.embed_query, OpenAIEmbeddings.embed_documents = embed_query, embed_documents

docs_db = (ROOT / "store" / "docs.db").resolve().as_posix()
_connect = sqlite3.connect
def connect(database, *a, **k):
    s = str(database).replace("\\", "/")
    if s.endswith("docs.db") or "docs.db?" in s:
        if "mode=ro" not in s:
            events["writes"].append(f"sqlite 쓰기 모드 연결: {s}")
    return _connect(database, *a, **k)
sqlite3.connect = connect

CHROMA_DB = ROOT / "store" / "chroma" / "chroma.sqlite3"
CHROMA_BOOKKEEPING = {"acquire_write"}   # 클라이언트를 열 때마다 chromadb가 쓰는 내부 잠금 기록 (색인 내용 아님)

def chroma_tables():
    c = _connect(f"file:{CHROMA_DB.as_posix()}?mode=ro", uri=True)
    try:
        names = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {n: c.execute(f'SELECT count(*) FROM "{n}"').fetchone()[0] for n in names}
    finally:
        c.close()

def snapshot():
    """store/ 파일의 크기·수정 시각. chroma.sqlite3는 수정 시각 대신 테이블별 행 수로 비교한다."""
    snap = {p.relative_to(ROOT).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in (ROOT / "store").rglob("*") if p.is_file()}
    key = CHROMA_DB.relative_to(ROOT).as_posix()
    if key in snap:
        snap[key] = (snap[key][0], tuple(sorted((k, v) for k, v in chroma_tables().items()
                                                 if k not in CHROMA_BOOKKEEPING)))
    return snap

chroma_before = chroma_tables()
chroma_mtime_before = CHROMA_DB.stat().st_mtime_ns

before = snapshot()
from streamlit.testing.v1 import AppTest
out = {}
at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=180).run()
out["tabs"] = [t.label for t in at.tabs]
out["startup_exceptions"] = [str(e.value)[:200] for e in at.exception]
out["startup_errors"] = [str(e.value)[:200] for e in at.error]

at.text_input(key="search_q").input("리가켐바이오 기술이전 계약 상대방")
at.button(key="search_btn").click().run()
md = " ".join(m.value for m in at.markdown)
out["search_exceptions"] = [str(e.value)[:200] for e in at.exception]
out["search_shows_results"] = "리가켐" in md or any("리가켐" in str(x.value) for x in at.expander)

out["search_errors"] = [str(e.value)[:200] for e in at.error]
out["chat_inputs_after_search"] = len(at.chat_input)
chat_md, out["chat_exceptions"] = "", []
try:
    at.chat_input[0].set_value("종근당이 2023년 2월 도입한 ADC 플랫폼 기술의 계약 상대방은?").run()
    for _ in range(3):           # 스트리밍·완료 처리에 rerun이 걸리는 경우
        at.run()
    chat_md = " ".join(m.value for m in at.chat_message[-1].markdown) if at.chat_message else ""
    out["chat_exceptions"] = [str(e.value)[:200] for e in at.exception]
except Exception as e:
    import traceback
    out["chat_exceptions"] = [f"{type(e).__name__}: {e}", traceback.format_exc()[-600:]]
out["chat_messages"] = len(at.chat_message)
out["chat_answer_head"] = chat_md[:160]

out["dashboard_rendered"] = any("실험 설정별" in s.value or "성능" in s.value for s in at.subheader) or \
    any("Hit@" in m.label for m in at.metric)

after = snapshot()
out["store_changed"] = sorted(k for k in after if before.get(k) != after[k])
out["store_new"] = sorted(set(after) - set(before))
chroma_after = chroma_tables()
out["chroma_internal_changes"] = {k: [chroma_before.get(k), chroma_after.get(k)] for k in set(chroma_before) | set(chroma_after)
                                  if chroma_before.get(k) != chroma_after.get(k)}
out["chroma_mtime_changed"] = CHROMA_DB.stat().st_mtime_ns != chroma_mtime_before
out["chroma_rows"] = {k: chroma_after[k] for k in ("embeddings", "collections", "segments") if k in chroma_after}
out.update(events)
print("RESULT " + json.dumps(out, ensure_ascii=False))
'''


def runtime_check() -> dict:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-c", RUNTIME, str(ROOT)], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=900, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
    if line is None:
        return {"error": (r.stderr or r.stdout)[-2000:]}
    return json.loads(line[len("RESULT "):])


def main() -> None:
    ap = argparse.ArgumentParser(description="계층 분리 검증")
    ap.add_argument("--static", action="store_true", help="정적 검사만 (API 호출 없음)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "layer_check.json"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rep = {"static": static_check()}
    ok = all(not v["violations"] for v in rep["static"].values())
    for layer, v in rep["static"].items():
        mark = "OK " if not v["violations"] else "FAIL"
        print(f"[정적 {mark}] {layer:9s} 파일 {v['files']:2d} · 프로젝트 내 의존 {v['project_imports']} — {v['rule']}")
        for x in v["violations"]:
            print("    -", x)
    if not args.static:
        rt = rep["runtime"] = runtime_check()
        if "error" in rt:
            ok = False
            print("[실행 FAIL] 앱 구동 실패\n", rt["error"])
        else:
            checks = {
                "탭 3개 구동": rt["tabs"] == ["검색", "대화", "성능 분석"] and not rt["startup_exceptions"]
                and not rt["startup_errors"],
                "검색 탭 결과 표시": rt["search_shows_results"] and not rt["search_exceptions"],
                "대화 탭 답변 생성": bool(rt["chat_answer_head"]) and not rt["chat_exceptions"],
                "성능 분석 탭 렌더링": rt["dashboard_rendered"],
                "collector·indexer import 시도 0건": not rt["blocked_imports"],
                "저장소 쓰기 시도 0건": not rt["writes"],
                "docs.db 변경 없음": not any(k.startswith("store/docs.db") for k in rt["store_changed"] + rt["store_new"]),
                "Chroma·BM25·청크 파일 변경 없음": not [k for k in rt["store_changed"] + rt["store_new"]
                                                if not k.startswith("store/docs.db")],
            }
            rt["checks"] = checks
            for name, passed in checks.items():
                print(f"[실행 {'OK ' if passed else 'FAIL'}] {name}")
                ok &= passed
            if rt["store_changed"] or rt["store_new"]:
                print("    store/ 변경:", rt["store_changed"], "신규:", rt["store_new"])
            print(f"    대화 답변 앞부분: {rt['chat_answer_head'][:100]}")
    rep["passed"] = ok
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'통과' if ok else '실패'} → {Path(args.out).relative_to(ROOT).as_posix()}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

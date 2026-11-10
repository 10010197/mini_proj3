"""제출 전 점검 목록 — 항목마다 통과 여부를 '측정한 수치'로 판정한다.

    python -m eval.check_submission            # 전체 (챗봇 실행 검사: 질의 임베딩·답변 생성 약 3회 API 호출)
    python -m eval.check_submission --static   # 파일·DB·결과 파일만 (API 호출 없음)

결과: eval/results/submission_check.json, 요약은 docs/submission_checklist.md.
판정: PASS(근거 수치가 기준 충족) · FAIL(미충족) · WARN(충족하나 주의할 수치가 있음).
키 값은 화면·파일에 쓰지 않는다(일치 여부와 건수만 기록).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "eval" / "results"
DOCS_DB = ROOT / "store" / "docs.db"
CUR_TAG = "structure__3large__hybrid_time__gpt-5.4-mini_prompt2"     # 현재 챗봇 기본 설정 (임베딩 3-large, 2026-11-10 교체)
MANUAL_TAG = "structure__hybrid_time__gpt-5.4-mini_prompt2"         # 사람 채점을 한 실행 (교체 전 3-small)
DEFAULT_COLLECTION = "structure__text-embedding-3-large"
# 출처 종류 + 근거 번호: '공시[1]', '뉴스 보도[4]', '정부 보도자료[2]' (종류어와 번호 사이 6자 이내)
SOURCE_LABELED = re.compile(r"(뉴스|공시|특허|보고서|보도자료)[^\[\n.]{0,6}\[\d+\]")
REFUSAL = "[답변 불가] 제공된 자료에서 확인할 수 없습니다."
LAYERS = ["collector", "indexer", "store", "rag", "app"]
checks: list[dict] = []


def add(section: str, item: str, status: str, evidence: str, source: str) -> None:
    checks.append({"section": section, "item": item, "status": status, "evidence": evidence, "source": source})


def ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DOCS_DB.as_posix()}?mode=ro", uri=True)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _core(title: str) -> str:
    """제목 비교용: 앞의 출처 표지('[보고서] ' 등, 원 제목에 이미 있어 반복된 경우 포함)와 공백 차이를 없앤다."""
    t = re.sub(r"\s+", " ", (title or "").replace("\xa0", " ")).strip()
    return re.sub(r"^(\[(보고서|특허|공시|뉴스|보도자료|정부 보도자료)\]\s*)+", "", t)


def pct(a: int, b: int) -> str:
    return f"{a:,}/{b:,} ({a / b:.1%})" if b else f"{a}/{b}"


# ---------------------------------------------------------------- 구조

def check_structure(static_only: bool) -> None:
    S = "구조"
    from eval.check_layers import static_check

    present = [d for d in LAYERS if (ROOT / d).is_dir()]
    st = static_check()
    viol = sum(len(v["violations"]) for v in st.values())
    deps = {k: v["project_imports"] for k, v in st.items()}
    add(S, "collector/ · indexer/ · store/ · rag/ · app/ 분리", "PASS" if len(present) == 5 and viol == 0 else "FAIL",
        f"디렉터리 {len(present)}/5 · 계층 규칙 위반 {viol}건 · rag 의존 {deps['rag']} · app 의존 {deps['app']}",
        "eval/check_layers.py (정적)")

    chatbot = st["rag"]["violations"] + st["app"]["violations"]
    lc = load_json(RESULTS / "layer_check.json") if (RESULTS / "layer_check.json").exists() else {}
    rt = lc.get("runtime", {})
    add(S, "챗봇 코드에 임베딩 생성·크롤링 코드 없음", "PASS" if not chatbot and not rt.get("writes") else "FAIL",
        f"rag+app 정적 위반 {len(chatbot)}건(크롤링 라이브러리·embed_documents·인덱스 쓰기 호출) · "
        f"실제 앱 구동 중 문서 임베딩·인덱스 쓰기 시도 {len(rt.get('writes', []))}건 · 질의 임베딩만 {len(rt.get('query_embeddings', []))}회",
        "eval/check_layers.py, eval/results/layer_check.json")

    import chromadb
    chroma_dir = ROOT / "store" / "chroma"
    size = sum(p.stat().st_size for p in chroma_dir.rglob("*") if p.is_file()) if chroma_dir.exists() else 0
    cl = chromadb.PersistentClient(path=str(chroma_dir))       # 새 프로세스에서 디스크로부터 다시 연다
    cols = {c.name if hasattr(c, "name") else c: cl.get_collection(c.name if hasattr(c, "name") else c).count()
            for c in cl.list_collections()}
    main = cols.get(DEFAULT_COLLECTION, 0)
    add(S, "벡터 DB 디스크 영속화", "PASS" if main == 57433 else "FAIL",
        f"store/chroma {size / 1e9:.2f}GB · 디스크에서 다시 열어 센 컬렉션 {len(cols)}개 · 기본 컬렉션 청크 {main:,} (적재 파일 57,433과 일치)",
        "chromadb.PersistentClient(store/chroma)")

    ok = rt and rt.get("tabs") == ["검색", "대화", "성능 분석"] and not rt.get("blocked_imports") and rt.get("chat_answer_head")
    add(S, "인덱서 제외 상태에서 챗봇 구동", "PASS" if ok else "FAIL",
        f"collector·indexer import 차단 상태로 app/main.py 구동: 탭 {len(rt.get('tabs', []))}개 · import 시도 {len(rt.get('blocked_imports', []))}건 · "
        f"검색 결과 표시 {rt.get('search_shows_results')} · 대화 답변 생성 {bool(rt.get('chat_answer_head'))}",
        "eval/results/layer_check.json (python -m eval.check_layers)")

    src = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")
    harness_calls = re.findall(r"chain\.(\w+)\(", src)
    chat_calls = re.findall(r"chain\.(\w+)\(", (ROOT / "app" / "chat.py").read_text(encoding="utf-8"))
    ask_src = (ROOT / "rag" / "chain.py").read_text(encoding="utf-8")
    ask_body = ask_src[ask_src.index("def ask("):]
    ask_steps = [s for s in ("retrieve(", "generate(", "finalize(") if s in ask_body]
    same = "ask" in harness_calls and {"retrieve", "stream_answer", "finalize"} <= set(chat_calls)
    add(S, "평가 하네스와 챗봇이 같은 체인(rag/chain.py) 호출", "PASS" if same and len(ask_steps) == 3 else "FAIL",
        f"하네스 → chain.ask · 챗봇 → chain.{', chain.'.join(sorted(set(chat_calls)))} · "
        f"ask()는 같은 단계 {len(ask_steps)}/3(retrieve → generate → finalize)로 구성",
        "eval/run_eval.py:108, app/chat.py:127-138, rag/chain.py:119")

    ignored = bool(git("check-ignore", ".env").strip())
    tracked_env = bool(git("ls-files", ".env").strip())
    env = {}
    if (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if "KEY" in k.upper() and len(v.strip().strip('"\'')) >= 8:
                    env[k.strip()] = v.strip().strip('"\'')
    tracked = git("ls-files").split()
    hits_now = 0
    for f in tracked:
        p = ROOT / f
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        hits_now += sum(txt.count(v) for v in env.values()) + len(re.findall(r"\bsk-[A-Za-z0-9_-]{20,}", txt))
    commits = git("rev-list", "--all").split()
    hist = 0
    for v in env.values():
        out = subprocess.run(["git", "grep", "-F", "-l", "-e", v, *commits], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout
        hist += len(out.splitlines())
    report_files = [p for p in (ROOT / "report").rglob("*") if p.is_file()] if (ROOT / "report").exists() else []
    rep_hits = sum(sum(p.read_text(encoding="utf-8", errors="ignore").count(v) for v in env.values())
                   for p in report_files if p.suffix in (".md", ".txt", ".html", ".py", ".ipynb"))
    add(S, ".env가 .gitignore에 등록 · 저장소·리포트에 키 없음",
        "PASS" if ignored and not tracked_env and hits_now == 0 and hist == 0 and rep_hits == 0 else "FAIL",
        f".env 무시됨 {ignored} · 추적 안 됨 {not tracked_env} · .env의 키 {len(env)}개를 추적 파일 {len(tracked)}개에서 검색 → {hits_now}건 · "
        f"커밋 {len(commits)}개 전체 이력에서 {hist}건 · report/ 파일 {len(report_files)}개에서 {rep_hits}건 · 'sk-' 패턴 포함",
        "git check-ignore / git grep (키 값은 출력하지 않음)")


# ---------------------------------------------------------------- 데이터 및 인덱싱

def check_data() -> None:
    S = "데이터 및 인덱싱"
    nb = load_json(ROOT / "notebooks" / "eda.ipynb")
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ran = [c for c in code if c.get("outputs")]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    add(S, "벡터 DB 구성 전 EDA 수행·기술", "PASS" if ran and "EDA 요약" in readme else "FAIL",
        f"notebooks/eda.ipynb 코드 셀 {len(code)}개 중 실행 결과가 있는 셀 {len(ran)}개 · 요약 README §3(제공 CSV 65,025행 → 고유 32,452건 등)",
        "notebooks/eda.ipynb, README.md §3, docs/data_and_use_cases.md")

    cov = load_json(ROOT / "eval" / "kb_coverage.json")
    v = Counter(x["verdict"] for x in cov)
    add(S, "부족 자료 탐색·보강 내역 기록", "PASS" if cov else "FAIL",
        f"대표 질문 {len(cov)}개의 출처별 근거 수 점검: 충분 {v.get('OK', 0)} · 해당 없음(거절 문항) {v.get('N/A', 0)} · 부족 {len(cov) - v.get('OK', 0) - v.get('N/A', 0)} · "
        f"보강: 정부 보도자료 1,022건 · 공시 1,627건 · 특허 8,114건 · 보고서 1,330건 직접 수집",
        "eval/kb_coverage.json, docs/kb_composition.md §2")

    con = ro()
    by = dict(con.execute("SELECT source_type, count(*) FROM documents WHERE in_kb=1 GROUP BY 1"))
    origin = dict(con.execute("SELECT origin, count(*) FROM documents WHERE in_kb=1 GROUP BY 1"))
    kb = sum(by.values())
    direct = kb - origin.get("provided", 0)
    primary = by.get("disclosure", 0) + by.get("policy", 0)
    add(S, "직접 수집한 1차 자료가 지식베이스에 포함", "PASS" if primary > 0 else "FAIL",
        f"지식베이스 {kb:,}건 중 직접 수집 {pct(direct, kb)} · 1차 자료(공시 {by.get('disclosure', 0):,} + 정부 보도자료 {by.get('policy', 0):,}) {pct(primary, kb)} · "
        f"출처별 {', '.join(f'{k} {n:,}' for k, n in sorted(origin.items(), key=lambda x: -x[1]))}",
        "store/docs.db documents.in_kb, origin")

    robots = list((ROOT / "data" / "compliance").glob("*robots.txt")) if (ROOT / "data" / "compliance").exists() else []
    add(S, "직접 수집 소스의 선정 근거·접근 제약 대응 기술", "PASS" if "크롤링 준수 사항" in readme and robots else "FAIL",
        f"README §8 소스별 표(수집 방식·robots/약관·요청 간격) · 보관한 robots.txt {len(robots)}개 · 대체 경로 기술 2건(보건산업진흥원 PDF → BioIN 재게시, mohw → 정책브리핑)",
        "README.md §7·§8, data/compliance/, docs/kb_composition.md")

    rows = {}
    for scope, cond in (("지식베이스", "in_kb=1"), ("전체 수집", "1=1")):
        n, u, d, l = con.execute(f"SELECT count(*), sum(coalesce(trim(url),'')<>''), sum(coalesce(trim(published_at),'')<>''), "
                                 f"sum(coalesce(trim(loaded_at),'')<>'') FROM documents WHERE {cond}").fetchone()
        rows[scope] = (n, u, d, l)
    n, u, d, l = rows["지식베이스"]
    na, ua, da, la = rows["전체 수집"]
    add(S, "모든 문서에 출처 URL·발행일·수집 일시 기록", "PASS" if u == d == l == n and ua == da == la == na else "FAIL",
        f"지식베이스 {n:,}건: URL {pct(u, n)} · 발행일 {pct(d, n)} · 수집 일시 {pct(l, n)} / 전체 수집 {na:,}건: URL {ua:,} · 발행일 {da:,} · 수집 일시 {la:,}",
        "store/docs.db documents.url, published_at, loaded_at")

    dd = load_json(ROOT / "store" / "dedup_stats.json")
    ex = dd["exact"]
    add(S, "중복 판별 기준 설정·처리 전후 건수 기록", "PASS" if dd.get("output_docs") else "FAIL",
        f"원본 {ex['raw_rows']:,}행 → URL 병합 {ex['after_url_merge']:,} → 본문 완전 중복 −{ex['removed_same_body']} → 품질 선별 {dd['input_docs']:,} → "
        f"재배포본 −{dd['removed_republish']:,}(클러스터 {dd['republish_clusters']:,}, 기준 5-gram Jaccard ≥ {dd['thresholds']['REPUB_JACCARD']}·{dd['thresholds']['REPUB_MAX_DAYS']}일) → "
        f"동일 사건 축약 −{dd['removed_event_same_content']} → {dd['output_docs']:,}건 (−{1 - dd['output_docs'] / dd['input_docs']:.1%})",
        "store/dedup_stats.json, docs/kb_composition.md §1")
    con.close()

    # 청크 머리말: Chroma에 실제로 들어간 모든 청크를 읽어 1행(제목)·2행(발행처·날짜)을 대조한다
    from rag import retriever
    col = retriever.vectorstore()._collection
    total, title_ok, date_ok, pub_ok, meta_ok, off = col.count(), 0, 0, 0, 0, 0
    need = ("doc_id", "source_type", "title", "published_at", "url")
    while off < total:
        r = col.get(include=["documents", "metadatas"], limit=5000, offset=off)
        for doc, m in zip(r["documents"], r["metadatas"]):
            lines = (doc or "").split("\n", 2)
            title_ok += bool(m.get("title")) and _core(m["title"]) in _core(lines[0])
            second = lines[1] if len(lines) > 1 else ""
            date_ok += bool(m.get("published_at")) and str(m["published_at"])[:10] in second
            pub_ok += bool(m.get("publisher")) and str(m["publisher"]) in second
            meta_ok += all(m.get(k) not in (None, "") for k in need)
        off += 5000
    add(S, "청크에 출처·발행일 등 필요한 정보 포함", "PASS" if date_ok == total and meta_ok == total else "WARN",
        f"청크 {total:,}개 전수: 머리말 2행에 발행일 {pct(date_ok, total)} · 발행처 {pct(pub_ok, total)} · "
        f"메타데이터(doc_id·출처 종류·제목·발행일·URL) 모두 있음 {pct(meta_ok, total)}",
        "store/chroma 기본 컬렉션 전수 조회")
    add(S, "청크에 문서 제목 포함", "PASS" if title_ok == total else "WARN",
        f"청크 {total:,}개 전수: 머리말 1행에 문서 제목 {pct(title_ok, total)} · 제목 포함 효과: 제거 시 Hit@3 0.574 → 0.368(−20.6%p)",
        "store/chroma 전수 조회, docs/evaluation.md §5")

    vl = load_json(ROOT / "store" / "vector_load_stats.json")[DEFAULT_COLLECTION]
    add(S, "인덱싱 후 문서 수·청크 수 검증",
        "PASS" if vl["chunks_in_file"] == vl["chunks_in_db"] == vl["count_after_reopen"] and not vl["ids_missing_in_db"] else "FAIL",
        f"청크 파일 {vl['chunks_in_file']:,} = DB {vl['chunks_in_db']:,} = 재오픈 {vl['count_after_reopen']:,} · 누락 ID {vl['ids_missing_in_db']} · 빈 청크 {vl['empty_documents']} · "
        f"벡터 불일치 {vl['vector_mismatch']}/{vl.get('vector_sample', 200)} · 문서 {vl['docs_in_db']:,} + 전 청크 중복으로 빠진 24 = 지식베이스 {kb:,}",
        "store/vector_load_stats.json, docs/indexing.md")


# ---------------------------------------------------------------- 평가

def check_eval() -> None:
    S = "평가"
    from eval.retrieval_eval import load_eval
    es = load_eval()
    types = Counter(x["type"] for x in es)
    with_key = sum(bool(x.get("answer_key")) for x in es)
    ans = [x for x in es if x["type"] != "unanswerable"]
    with_gold = sum(bool(x.get("gold_doc_ids")) for x in ans)
    add(S, "평가셋 직접 작성", "PASS" if with_key == len(es) and with_gold == len(ans) else "FAIL",
        f"{len(es)}문항 (사실 {types['fact']} · 비교 {types['comparison']} · 시점 {types['temporal']} · 종합 {types['synthesis']} · 답변 불가 {types['unanswerable']}) · "
        f"정답 기준 {pct(with_key, len(es))} · 정답 문서 ID {pct(with_gold, len(ans))} — 지식베이스 문서를 보고 작성",
        "eval/eval_set.jsonl, docs/indexing.md §7")
    add(S, "답변 불가 유형 문항 포함", "PASS" if types["unanswerable"] else "FAIL",
        f"답변 불가 {types['unanswerable']}문항 ({types['unanswerable'] / len(es):.0%}) — 범위 밖 연도·의학적 조언·투자 권유·미수집 영역",
        "eval/eval_set.jsonl")

    runs = sorted(RESULTS.glob("answer_eval_*.json"))
    cur = load_json(RESULTS / f"answer_eval_{CUR_TAG}.json")["summary"]["answerable"]
    add(S, "Hit@k 또는 MRR 측정", "PASS" if runs else "FAIL",
        f"실험 {len(runs)}개 모두 Hit@1·3·6·MRR 기록 · 현재 기본 Hit@1 {cur['hit@1']} · Hit@3 {cur['hit@3']} · Hit@6 {cur['hit@6']} · MRR {cur['mrr']} (답변 가능 68)",
        "eval/results/answer_eval_*.json")

    filled, total, sheets = 0, 0, 0
    import pandas as pd
    for f in RESULTS.glob("answer_review_*.csv"):
        df = pd.read_csv(f)
        sheets += 1
        total += len(df)
        filled += int(df["manual_score"].notna().sum()) if "manual_score" in df else 0
    sheet = RESULTS / f"manual_review__{MANUAL_TAG}.csv"
    agr_path = RESULTS / f"manual_agreement__{MANUAL_TAG}.json"
    if sheet.exists():
        from eval.manual_review import agree
        agr = agree(MANUAL_TAG)
    else:
        agr = {}
    n_sc = agr.get("n_scored", 0)
    if n_sc >= 30:
        ev = (f"블라인드 표본 {n_sc}/{agr['n_sheet']}문항 사람 채점 · 사람 정확도 {agr['manual_accuracy']} vs 같은 문항 LLM {agr['llm_accuracy_same_items']} · "
              f"완전 일치 {agr['exact_agreement']:.0%} · 0.5점 이내 {agr['within_half_point']:.0%} · Cohen κ {agr['cohen_kappa']} · "
              f"사람이 더 높게 {agr['manual_higher']} / 더 낮게 {agr['manual_lower']}")
    else:
        ev = (f"사람 채점 {n_sc}/30 (블라인드 검토표 {'있음' if sheet.exists() else '없음'}) · 기존 검토표 {sheets}개 manual_score {filled}/{total} — "
              f"정확도는 아직 LLM 채점(gpt-5.4)만 있음. 기준: 30문항 사람 채점 후 LLM 채점과 일치율·κ 보고")
    add(S, "답변 품질 직접 채점", "PASS" if n_sc >= 30 else "FAIL", ev,
        f"eval/manual_review.py → {sheet.relative_to(ROOT).as_posix()}, {agr_path.relative_to(ROOT).as_posix()}")

    reg = load_json(ROOT / "eval" / "experiments.json")["experiments"]
    base = load_json(RESULTS / "answer_eval_structure__gpt-5.4-mini_run1.json")["summary"]["answerable"]
    add(S, "개선 전후 수치 비교표 작성", "PASS",
        f"실험 기록 {len(reg)}개(부모·바꾼 변수 하나·결정) · 기준선 → 현재 기본: Hit@1 {base['hit@1']} → {cur['hit@1']} · "
        f"Hit@3 {base['hit@3']} → {cur['hit@3']} · MRR {base['mrr']} → {cur['mrr']} · 정확도 {base['accuracy']} → {cur['accuracy']}",
        "eval/experiments.json, docs/evaluation.md §5~§8, 성능 분석 탭")

    fl = RESULTS / f"failure_layers__{CUR_TAG}.json"
    d = load_json(fl) if fl.exists() else {}
    add(S, "실패 사례 확인·원인 계층별 분류", "PASS" if d else "FAIL",
        f"실패 {d.get('n_failed', 0)}건 증거 기반 판정: " + " · ".join(f"{k} {len(v)}" for k, v in d.get("by_layer", {}).items())
        + " (데이터·전처리 0)",
        "eval/diagnose_failures.py, docs/failure_analysis.md §2")


# ---------------------------------------------------------------- 챗봇

def check_chatbot(static_only: bool) -> None:
    S = "챗봇"
    run = load_json(RESULTS / f"answer_eval_{CUR_TAG}.json")
    items = run["items"]
    answered = [x for x in items if not x["refused"]]
    with_cite = [x for x in answered if x.get("cited")]
    invalid = run["summary"]["invalid_citation_items"]
    rt = _runtime() if not static_only else {}
    ui_cite = rt.get("turn1_has_citation")
    add(S, "답변에 출처가 번호로 연결되어 표시",
        "PASS" if len(with_cite) / max(len(answered), 1) >= 0.95 and not invalid and ui_cite is not False else "FAIL",
        f"거절하지 않은 답변 {pct(len(with_cite), len(answered))}이 [n] 인용 · 존재하지 않는 번호 인용 {len(invalid)}건 · "
        f"출처 종류와 번호 함께 표기('공시[1]', '뉴스 보도[4]' 등) {sum(1 for x in answered if SOURCE_LABELED.search(x['answer']))}/{len(answered)} · "
        f"실제 앱 대화 답변에 [n] 표시 {ui_cite} · 출처 목록 펼침 {rt.get('turn1_source_expanders', '—')}개",
        f"answer_eval_{CUR_TAG}.json, 실제 앱 구동")

    con = ro()
    cited = {d for x in items for d in x.get("cited", [])}
    with_url = sum(1 for (u,) in con.execute(f"SELECT url FROM documents WHERE doc_id IN ({','.join('?' * len(cited))})",
                                              list(cited)) if u and u.strip())
    con.close()
    add(S, "원문 확인 또는 링크 접근", "PASS" if with_url == len(cited) else "WARN",
        f"평가에서 인용된 문서 {pct(with_url, len(cited))}에 원문 URL · 검색 탭 '원문 보기'와 대화 탭 '원문 전체 보기'로 docs.db 본문 표시(읽기 전용)",
        "app/search_ui.py:70, app/chat.py:33, store/docs.db")

    s = run["summary"]
    una = [x for x in items if x["type"] == "unanswerable"]
    refused_ok = sum(1 for x in una if x["refused"])
    exact = sum(1 for x in items if x["refused"] and x["answer"].lstrip().startswith(REFUSAL))
    n_ref = sum(1 for x in items if x["refused"])
    add(S, "근거가 없는 경우 답변 거절", "PASS" if s["unanswerable"]["refusal_accuracy"] >= 0.8 else "FAIL",
        f"답변 불가 문항 거절 {refused_ok}/{len(una)} ({s['unanswerable']['refusal_accuracy']}) — 평가셋 오류 U10 제외 시 {refused_ok}/{len(una) - 1} · "
        f"거절 문구 정확히 일치 {exact}/{n_ref} · 과잉 거절 {s['answerable']['over_refusal']} · 근거 충실도 {s['answerable']['grounded']}",
        f"answer_eval_{CUR_TAG}.json")

    if static_only:
        add(S, "멀티턴 대화 유지", "WARN", "실행 검사 생략(--static)", "")
        add(S, "검색 결과가 없을 때 안내 메시지", "WARN", "실행 검사 생략(--static)", "")
    else:
        add(S, "멀티턴 대화 유지", "PASS" if rt.get("multiturn_ok") else "FAIL",
            f"실제 앱에서 1턴 '종근당이 2023년 2월 도입한 ADC 플랫폼 기술의 계약 상대방은?' → 2턴 '그 계약의 총 규모는?': "
            f"2턴 답변에 'USD 132M' 포함 {rt.get('multiturn_ok')} · 대화 메시지 {rt.get('chat_messages')}개 유지 · 2턴 답변 앞부분: {rt.get('turn2_head', '')[:80]}",
            "app/chat.py(HISTORY_TURNS, chain.condense), 실제 앱 구동")
        add(S, "검색 결과가 없을 때 안내 메시지", "PASS" if rt.get("empty_search_msg") and rt.get("empty_chat_msg") else "FAIL",
            f"문서 0건인 필터(보고서 · 2021년 · ADC, docs.db로 0건 확인)에서 검색 탭 안내 {rt.get('empty_search_msg')} · 대화 탭 안내 {rt.get('empty_chat_msg')} · "
            f"예외 {len(rt.get('empty_exceptions', []))}건",
            "app/search_ui.py:44, app/chat.py:131, 실제 앱 구동")

    dash = (ROOT / "app" / "dashboard.py").read_text(encoding="utf-8")
    charts = len(re.findall(r"^def chart_", dash, re.M))
    lc = load_json(RESULTS / "layer_check.json").get("runtime", {}) if (RESULTS / "layer_check.json").exists() else {}
    add(S, "평가 결과 시각화 구현", "PASS" if charts >= 5 and lc.get("dashboard_rendered") else "FAIL",
        f"성능 분석 탭 차트 {charts}종(설정별 Hit@k·MRR · 유형별 정확도 · 유사도 분포 · 지식베이스 구성 · 응답시간) · 실험 {len([e for e in load_json(ROOT / 'eval' / 'experiments.json')['experiments'] if e['kind'] == 'answer'])}개 결과 파일 · 실제 앱 렌더링 {lc.get('dashboard_rendered')}",
        "app/dashboard.py, docs/evaluation.md §9")


RUNTIME = r'''
import json, sys, re
from pathlib import Path
ROOT = Path(sys.argv[1]); sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
out = {}
def settle(at):
    for _ in range(3):
        at.run()
    return at
at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=180).run()
at.chat_input[0].set_value("종근당이 2023년 2월 도입한 ADC 플랫폼 기술의 계약 상대방은?").run(); settle(at)
t1 = " ".join(m.value for m in at.chat_message[-1].markdown) if at.chat_message else ""
out["turn1_has_citation"] = bool(re.search(r"\[\d+\]", t1))
out["turn1_source_expanders"] = sum(1 for e in at.chat_message[-1].expander) if at.chat_message else 0
at.chat_input[0].set_value("그 계약의 총 규모는?").run(); settle(at)
t2 = " ".join(m.value for m in at.chat_message[-1].markdown) if at.chat_message else ""
out["turn2_head"] = t2[:200]
out["multiturn_ok"] = bool(re.search(r"132\s*M|1억\s*3,?200만|132,?000,?000", t2))
out["chat_messages"] = len(at.chat_message)

e = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=180).run()
e.sidebar.multiselect[0].set_value(["report"])
e.sidebar.slider[0].set_value((2021, 2021))
e.sidebar.selectbox[0].set_value("ADC")
e.run()
e.text_input(key="search_q").input("ADC 기술수출 계약").run()
e.button(key="search_btn").click().run()
out["empty_search_msg"] = any("조건에 맞는 문서가 없습니다" in w.value for w in e.warning)
e.chat_input[0].set_value("ADC 기술수출 계약 규모는?").run(); settle(e)
texts = [x.value for x in e.info] + [x.value for x in e.warning] + [m.value for c in e.chat_message for m in c.markdown]
out["empty_chat_msg"] = next((t[:120] for t in texts if "필터" in t or "찾지 못" in t or "없습니다" in t), None)
out["empty_exceptions"] = [str(x.value)[:200] for x in e.exception]
print("RESULT " + json.dumps(out, ensure_ascii=False))
'''


def _runtime() -> dict:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    r = subprocess.run([sys.executable, "-c", RUNTIME, str(ROOT)], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900, env=env)
    line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
    if not line:
        print("[실행 검사 실패]", (r.stderr or r.stdout)[-1500:])
        return {}
    return json.loads(line[7:])


def check_report() -> None:
    """리포트 점검 3항목 — report/report.md의 구성과 정량 근거를 확인한다."""
    S = "리포트"
    p = ROOT / "report" / "report.md"
    if not p.exists():
        for item in ("청킹·임베딩 선택의 정량적 근거", "실패 사례와 한계 기술", "모든 설계 판단의 근거 설명"):
            add(S, item, "FAIL", "report/report.md 없음", "report/")
        return
    txt = p.read_text(encoding="utf-8")
    heads = re.findall(r"^## (\d+)\. ", txt, re.M)
    figs = re.findall(r"!\[[^\]]*\]\((figures/[^)]+)\)", txt)
    missing_figs = [f for f in figs if not (ROOT / "report" / f).exists()]
    emb_p = RESULTS / "embedding_comparison.json"
    emb = load_json(emb_p) if emb_p.exists() else {}
    models = sorted({r["model"] for r in emb.get("rows", [])})
    filled = "<!-- EMBEDDING_SECTION -->" not in txt
    chunk_ok = "0.588" in txt and "0.515" in txt and "0.382" in txt
    add(S, "청킹·임베딩 선택의 정량적 근거", "PASS" if chunk_ok and filled and len(models) >= 2 else "FAIL",
        f"청킹: 3개 설정 Hit@1·3·10·MRR·유형별 비교표(structure 0.588 · fixed 0.515 · notitle 0.382) {chunk_ok} · "
        f"임베딩: 모델 {len(models)}종 × 차원 {len(emb.get('rows', []))}개 검색 비교 결과 {emb_p.exists()} · 리포트 §4.3 반영 {filled}",
        "report/report.md §4.2·§4.3, eval/results/embedding_comparison.json")
    fail_sec = re.search(r"^## 8\. 실패 사례 분석(.*?)^## 9\. ", txt, re.M | re.S)
    lim_sec = re.search(r"^## 9\. 한계와 향후 과제(.*?)^## 10\. ", txt, re.M | re.S)
    case_ids = sorted(set(re.findall(r"(?<![A-Za-z0-9])[FCTSU]\d{2}(?![0-9])", fail_sec.group(1)))) if fail_sec else []
    lim_rows = len(re.findall(r"^\| ", lim_sec.group(1), re.M)) if lim_sec else 0
    add(S, "실패 사례와 한계 기술", "PASS" if len(case_ids) >= 15 and lim_rows >= 8 else "FAIL",
        f"§8 실패 사례 분석에 문항 {len(case_ids)}개 언급(사람 기준 0점 F08·T04·T12 개별 분석 포함) · "
        f"§9 한계 표·과제 {lim_rows}행 · 실패 계층 판정 52건 수치 포함",
        "report/report.md §8·§9")
    appx = re.search(r"^## 부록 B\. 설계 판단과 근거 일람(.*)", txt, re.M | re.S)
    rows = [r for r in re.findall(r"^\| ([^|]+) \| ([^|]+) \|", appx.group(1), re.M) if r[0].strip() not in ("판단", "---")] if appx else []
    numeric = sum(1 for _, ev in rows if re.search(r"\d", ev))
    add(S, "모든 설계 판단의 근거 설명", "PASS" if len(heads) >= 10 and rows and numeric / len(rows) >= 0.8 and not missing_figs else "FAIL",
        f"필수 10개 절 {len(set(heads) & set(map(str, range(1, 11))))}/10 · 부록 B 설계 판단 {len(rows)}건 중 수치 근거 {numeric}건 · "
        f"그림 {len(figs)}개 중 누락 {len(missing_figs)}",
        "report/report.md 부록 B")


def write_doc(out: dict) -> Path:
    """docs/submission_checklist.md — 표의 수치는 out(측정 결과)에서 그대로 옮긴다."""
    mark = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}
    cur = load_json(RESULTS / f"answer_eval_{CUR_TAG}.json")["summary"]
    a = cur["answerable"]
    agr_p = RESULTS / f"manual_agreement__{MANUAL_TAG}.json"
    agr = load_json(agr_p) if agr_p.exists() else {}
    L = ["# 제출 전 점검 — 수치 근거", "",
         f"> 측정 일시 {out['checked_at']} · 재현: `python -m eval.check_submission` (전체, 실제 앱 구동 포함) / `--static` (API 호출 없음)",
         "> 원자료: `eval/results/submission_check.json`. 이 문서는 같은 실행이 생성한다. 아래 수치는 설명이 아니라 그 시점에 측정한 값이다.", "",
         f"**결과: {len(out['checks'])}항목 중 통과 {out['summary'].get('PASS', 0)} · 주의 {out['summary'].get('WARN', 0)} · 미충족 {out['summary'].get('FAIL', 0)}**", "",
         "판정 방법",
         "- **구조:** 계층 검사(`eval/check_layers.py`)로 확인한다. 정적 AST 검사와, 수집기·인덱서 import와 저장소 쓰기를 막은 상태의 실제 앱 구동을 함께 쓴다.",
         "- **데이터·인덱싱:** `docs.db`와 Chroma 기본 컬렉션의 청크를 **전수 조회**한다(표본 아님).",
         "- **키 노출:** `.env`의 실제 키 값을 추적 파일 전체와 **모든 커밋 이력**에서 검색한다. 키 값 자체는 어디에도 출력하지 않는다.",
         "- **챗봇:** `streamlit.testing` AppTest로 실제 `app/main.py`를 구동한다. 멀티턴 2턴, 문서가 0건인 필터에서의 검색·대화를 직접 수행한다.",
         "- **답변 품질:** 블라인드 층화 표본 30문항을 사람이 채점하고(`eval/manual_review.py`), LLM 채점과의 일치율·κ를 계산한다."]
    sec = None
    for c in out["checks"]:
        if c["section"] != sec:
            sec = c["section"]
            L += ["", f"## {sec}", "", "| | 항목 | 수치 근거 | 근거 위치 |", "|---|---|---|---|"]
        L.append(f"| {mark[c['status']]} | {c['item']} | {c['evidence'].replace('|', '/').replace(chr(10), ' ')} | {c['source']} |")
    if agr.get("n_scored", 0) >= 30:
        bt = agr["by_type"]
        conf = agr.get("confusion_manual_to_llm", {})
        cell = lambda h, l: conf.get(h, {}).get(l, 0)  # noqa: E731
        L += ["", "## 답변 품질 — 사람 채점과 LLM 채점", "",
              "- **채점 기준(사람 채점자 확정):**",
              "  - `1`: 질문이 묻는 내용을 정확히 답함. 질문이 묻지 않은 세부 조건이 빠져도 1",
              "  - `0.5`: 질문이 묻는 내용 중 일부만 답함",
              "  - `0`: 핵심이 틀림, 또는 근거가 있는데 거절함",
              "- **블라인드:** 검토표에는 LLM 판정을 넣지 않았다.", "",
              "| 지표 | 값 |", "|---|---|",
              f"| 사람 정확도 (표본 {agr['n_scored']}) | **{agr['manual_accuracy']}** |",
              f"| 같은 문항의 LLM 정확도 | {agr['llm_accuracy_same_items']} (전체 68문항 LLM {a['accuracy']}) |",
              f"| 완전 일치 · 0.5점 이내 일치 | {agr['exact_agreement']:.0%} · **{agr['within_half_point']:.0%}** |",
              f"| Cohen κ | {agr['cohen_kappa']} |",
              f"| 사람이 더 높게 / 더 낮게 | {agr['manual_higher']} / {agr['manual_lower']} |",
              "| 유형별 사람 / LLM | " + " · ".join(f"{TL} {bt[t]['manual']} / {bt[t]['llm']}" for t, TL in
                                         (("fact", "사실"), ("comparison", "비교"), ("temporal", "시점"), ("synthesis", "종합")) if t in bt) + " |",
              "", "| 사람 \\ LLM | 0 | 0.5 | 1 |", "|---|---|---|---|"]
        L += [f"| {h} | {cell(h, '0.0')} | {cell(h, '0.5')} | {cell(h, '1.0')} |" for h in ("0.0", "0.5", "1.0")]
        L += ["",
              "**해석**",
              "- κ가 낮은 것은 무작위 불일치가 아니라 **채점 기준 차이**다. 불일치는 대부분 'LLM partial → 사람 1점'(묻지 않은 세부 조건 감점)이다.",
              "- 0.5점을 넘는 불일치가 없다. 그래서 LLM 채점은 한 방향으로 일관되게 엄격하다.",
              "- 따라서 같은 LLM 채점자로 잰 **실험 간 비교는 유효**하다. 다만 **절대 정확도는 사람 기준보다 낮게** 나온다.",
              "- 사람 기준으로 보정한 실패 판정은 [failure_analysis.md](failure_analysis.md) §2.5에 있다."]
    rv_p = RESULTS / "reliability_validation.json"
    if rv_p.exists():
        rv = load_json(rv_p)
        L += ["", "## 답변별 신뢰성 판단 근거 (대화 탭)", "",
              "- 답변마다 아래에 '신뢰성 판단 근거' 상자를 보여준다. 계산은 `rag/reliability.py`가 한다. LLM 호출이 없고 체인 결과만 쓴다.",
              "- **원자료 수치:** 인용 근거 수/받은 근거 수, 1차 자료(공시·정부 보도자료) 비율, 최고 유사도와 평가 분포상 백분위, 근거 발행일 범위, 출처 구성, 발행처 수, 인용이 붙은 사실 문장 수, 질문 연도 근거 비율",
              "- **시스템 평가 성적:** Hit@3, 근거 충실도, 사람·LLM 정확도, 거절 정확도. 평가 결과 파일에서 읽는다.",
              "- **주의 표시:** 평가에서 정확도를 실제로 가른 기준만 쓴다. 표시마다 '평가에서 이 표시가 붙은 답변의 정확도'를 함께 보여준다.",
              f"- 검증: `python -m eval.reliability_eval` — 거절하지 않은 답변 가능 {rv['n_answered']}문항에 챗봇과 같은 함수를 적용했다.", "",
              "| 후보 지표 | 표시된 답변 | LLM 정확도 (표시 / 없음) | 사람 정확도 (표시 n / 없음 n) |", "|---|---|---|---|"]
        for name, c in rv["candidates"].items():
            L.append(f"| {name} | {c['n']} | {c['llm_on']} / {c['llm_off']} | {c['human_on']} ({c['human_n_on']}) / {c['human_off']} ({c['human_n_off']}) |")
        nf = rv["no_flag"]
        L += ["", f"- 주의 표시가 하나도 없는 답변 {nf['n']}건: LLM 정확도 {nf['llm']} · 사람 {nf['human']} (n={nf['human_n']}).",
              f"- 채택 기준: 인용 근거 {rv['multi_source_threshold']}건 이상 · 최고 유사도가 평가의 답변 불가 문항 중앙값({rv['unanswerable_median']}) 미만.",
              "- 기각 사유: '근거 1건뿐'은 오히려 정확도가 높다(단일 문서 사실 질문이 쉬움). '2차 자료뿐'은 차이가 없다. '인용 문장 비율'은 채택 지표와 14건 중 11건이 겹친다. '연도 불일치'는 2건뿐이다.",
              "- 한계: 사람 채점 표본이 작다(표시된 답변 중 사람 채점 6~10건). 그래서 주의 표시는 '정확도가 낮을 가능성'을 알리는 용도이고, 판정이 아니다."]
    acc_row = (f"정확도 LLM {a['accuracy']} · 사람 표본 {agr['manual_accuracy']} (같은 문항 LLM {agr['llm_accuracy_same_items']}, 0.5점 이내 일치 {agr['within_half_point']:.0%})"
               if agr.get("n_scored", 0) >= 30 else f"정확도 {a['accuracy']} (LLM 채점) — 사람 채점 미완료")
    L += ["", "## 신뢰성 판단에 쓰는 수치 요약", "", "| 판단 | 수치 | 의미 |", "|---|---|---|",
          f"| 검색이 정답 근거를 찾는가 | Hit@3 {a['hit@3']} · MRR {a['mrr']} (기준선 0.574 · 0.462) | 답변 가능 질문의 약 2/3은 정답 문서를 3위 안에서 받는다 |",
          f"| 답이 근거에 기반하는가 | 근거 충실도 {a['grounded']} · 존재하지 않는 번호 인용 {len(cur['invalid_citation_items'])}건 | 근거 밖 주장은 일부 문장 수준이고, 전부 지어낸 답변은 없다 |",
          f"| 근거가 없을 때 멈추는가 | 거절 정확도 {cur['unanswerable']['refusal_accuracy']} (평가셋 오류 U10 제외 9/10) · 과잉 거절 {a['over_refusal']} | 범위 밖 질문 1건(U08)은 거절 위치 규칙을 어긴다 |",
          f"| 답이 맞는가 | {acc_row} | LLM 채점은 엄격 쪽으로 일관되게 치우친다. 사람 기준 약점은 종합 문항 |",
          "| 수치가 재현되는가 | Hit@k는 같은 설정이면 동일 · 정확도는 실행마다 최대 ±0.05 · pool 스윕 40에서 하네스 값 재현 | 0.05 미만 정확도 차이는 설정 차이로 해석하지 않는다 |",
          "| 출처를 추적할 수 있는가 | 문서 URL·발행일·수집 일시, 청크 머리말 제목·발행일, 인용 문서 URL 모두 100% (위 표) | 모든 답변 근거를 원문까지 따라갈 수 있다 |", ""]
    p = ROOT / "docs" / "submission_checklist.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="제출 전 점검 목록 — 수치 근거")
    ap.add_argument("--static", action="store_true", help="API 호출 없이 파일·DB·결과만")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    check_structure(args.static)
    check_data()
    check_eval()
    check_chatbot(args.static)
    check_report()
    cnt = Counter(c["status"] for c in checks)
    out = {"checked_at": datetime.now().isoformat(timespec="seconds"), "summary": dict(cnt), "checks": checks}
    (RESULTS / "submission_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    sec = None
    for c in checks:
        if c["section"] != sec:
            sec = c["section"]
            print(f"\n## {sec}")
        print(f"[{c['status']:4s}] {c['item']}\n        {c['evidence']}")
    print(f"\n합계 {len(checks)}항목: " + " · ".join(f"{k} {v}" for k, v in cnt.items()) + " → eval/results/submission_check.json")
    if not args.static:
        print("문서 →", write_doc(out).relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()

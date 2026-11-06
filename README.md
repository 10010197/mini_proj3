# mini_proj3 — 바이오·헬스케어 산업·기술 동향 RAG

> **주제 B. 산업·기술 동향** · 선택 산업: **바이오·헬스케어**
>
> 뉴스(현재), 특허(기술 선행 지표), 투자보고서(시장 전망)를 **서로 다른 시간축의 근거**로 함께 검색하고,
> 출처 종류에 따라 내용이 상충할 때 그 차이를 드러내는 답변을 구성하는 RAG 챗봇.

## 1. 개요

### 목표 사용자와 사용 시나리오
> 상세: [docs/data_and_use_cases.md](docs/data_and_use_cases.md) — 제공 데이터 구성·품질 진단, 사용자 4종, 업무 상황 5종, 대표 질문 17개

- **사용자**: 산업 분석가(주 사용자), 제약·바이오 BD·전략 담당, 투자 심사역, 정책·사업 기획자
- **업무 상황**: 동향 브리핑 작성 · 기술 성숙도 판단(뉴스 vs 특허) · 시장 전망 수치 검증(뉴스 vs 보고서) · 기업·딜 사실 확인(공시) · 규제 영향 파악
- **범위 밖**: 종목 추천, 의학적 판단, 지식베이스 기간(2021-10 ~ 2026-11) 밖의 사건

예시 질문
- "최근 2년간 ADC 기술수출 뉴스가 늘었는데, 국내 ADC 특허 출원도 같이 늘었어?"
- "글로벌 CDMO 시장 규모 전망이 기사와 보고서에서 어떻게 달라?"
- "리가켐바이오의 최근 기술이전 계약 상대방과 계약 규모는?"

### 출처별 시간축 (evidence_role)

| 출처 종류 (`source_type`) | 근거 역할 (`evidence_role`) | 의미 | 확보 방법 |
|---|---|---|---|
| `news` | `current` | 현재 시점의 사건·발표 | 제공 CSV + 크롤링 보강 |
| `disclosure` | `current` (실적 전망 공시는 `outlook`) | 기업 단위 1차 근거 (계약·임상·허가·투자) | **DART OpenAPI** ✅ |
| `patent` | `leading` | 기술 선행 지표 | **KIPRIS Plus OpenAPI** ✅ |
| `report` | `outlook` (전망 서술 없으면 `current`) | 시장 전망·산업 분석 | **BioIN 크롤링** ✅ (KOTRA·보건산업진흥원·바이오협회 브리프 포함) |

답변은 출처 종류별로 근거를 나눠 제시하고, 같은 질문에 대해 시간축이 다른 근거가 엇갈리면 **상충 사실과 각 근거의 시점·출처를 함께** 보여주는 것을 원칙으로 한다.

### 산업 키워드 (과제 제공)

| 핵심 | 기술 하위어 | 시장·정책 |
|---|---|---|
| `바이오 의약품` `디지털 헬스케어` | `ADC` `CDMO` `AI 신약개발` | `바이오 투자` `신약 기술수출` `의료 AI 규제` |

## 2. 아키텍처

| 구분 | ① 수집기 | ② 인덱서 | ③ VectorDB·원문 | ④ 챗봇 |
|---|---|---|---|---|
| 역할 | 제공 데이터 적재, 크롤링 | 파싱·청킹·임베딩 후 적재 | 저장 및 검색 제공 | 검색, 생성, UI |
| 실행 방식 | 배치 | 배치 | 상시 존재 | 상시 실행 |
| 금지 사항 | 차트·UI 코드 포함 | — | — | 임베딩 생성, 크롤링, 인덱스 쓰기 |

```
project3/
├── README.md                     # 개요, 실행 순서, 스크린샷
├── .env.example                  # 키 이름만 포함
├── requirements.txt
├── data/provided/                # 제공 CSV 원본 (git 제외)
├── notebooks/
│   └── eda.ipynb                 # 문서 탐색(EDA)
├── collector/                    # ① 수집 계층
│   ├── load_provided.py          #   제공 데이터 적재 및 선별  ✅
│   ├── crawl_dart.py             #   DART 공시 수집 (OpenAPI)  ✅
│   ├── crawl_kipris.py           #   KIPRIS Plus 특허 수집 (OpenAPI)  ✅
│   ├── crawl_reports.py          #   BioIN 산업보고서 크롤링  ✅
│   ├── crawl_policy.py           #   정부 보도자료 (정책브리핑 korea.kr)  ✅
│   ├── dedup_news.py             #   뉴스 3단계 중복 처리  ✅
│   ├── (crawl_blog.py)           #   블로그 — 수집하지 않음 (근거: docs/kb_composition.md §2.1)
│   ├── select_kb.py              #   지식베이스 선별 (in_kb)  ✅
│   └── storage.py                #   원문 저장 (공통 documents 스키마)  ✅
├── indexer/                      # ② 인덱싱 계층
│   ├── parse.py                  #   문서 종류별 섹션 분할·정제  ✅
│   ├── dedup.py                  #   청크 단위 중복 처리  ✅
│   ├── chunk.py                  #   청킹 설정 3종 + 청크 메타데이터  ✅
│   ├── embed.py                  #   임베딩 (text-embedding-3-small) + 캐시  ✅
│   ├── load.py                   #   Chroma 적재 + 적재 검증  ✅
│   └── bm25.py                   #   BM25 키워드 인덱스 (하이브리드 검색용)  ✅
├── store/                        # ③ 저장 계층 (chroma/ 벡터·청크·메타데이터, docs.db 원문)  ✅
├── rag/                          # ④ 체인 — 챗봇·평가 하네스가 같은 rag.chain.ask() 호출
│   ├── retriever.py              #   검색: 벡터(Chroma) + BM25 + 질문 연도 → RRF 결합, 문서 단위 정리, 원문 조회  ✅
│   ├── bm25.py                   #   BM25 읽기 전용 검색 + 공용 토크나이저  ✅
│   ├── rerank.py                 #   LLM 재정렬 (검색 방식 hybrid_rerank)  ✅
│   ├── prompts.py                #   시스템 프롬프트 (근거 번호 인용, 출처별 구분, 거절 규칙)  ✅
│   └── chain.py                  #   LangChain 체인 (ChatOpenAI, 기본 gpt-5.4-mini)  ✅
├── eval/                         # 평가 하네스
│   ├── run_eval.py               #   평가셋 전체를 rag.chain.ask로 실행 → 검색·인용·거절·정확도·충실도  ✅
│   ├── eval_set.jsonl            #   평가셋 (사실·비교·시점·종합·답변 불가)  ✅
│   ├── retrieval_eval.py         #   검색 평가 Hit@k · MRR  ✅
│   └── kb_coverage.py            #   대표 질문별 근거 커버리지 (부족 자료 탐색)  ✅
├── app/                          # ④ 챗봇 UI (Streamlit, 조회 전용 — rag/만 호출)
│   ├── main.py                   #   진입점: 사이드바 필터(출처·연도·주제·근거 수) + 검색/대화 탭  ✅
│   ├── search_ui.py              #   검색 탭: 상위 문서·청크·유사도·메타데이터, 원문 보기(docs.db)  ✅
│   ├── chat.py                   #   대화 탭: 검색 진행 표시 → 스트리밍 답변 + 출처·원문 일부, 멀티턴(후속 질문 재작성), 빈 결과 안내  ✅
│   └── dashboard.py              #   (예정) 평가 결과 시각화
├── docs/                         # 설계·근거 문서 (data_and_use_cases · kb_composition · indexing · evaluation)
└── report/
```

## 3. 제공 데이터

네이버 뉴스 기반 CSV 3종 (2021-10-16 ~ 2026-10-16). 컬럼: `제목, 본문전문, 원문URL, 발행일, 매체명, 문서종류, 수집키워드`

| 파일 | 수집 키워드 | 행 수 |
|---|---|---|
| `04_바이오헬스_키워드_바이오의약품_6580건_…csv` | 바이오 의약품 | 6,580 |
| `04_바이오헬스_키워드_디지털헬스케어_25991건_…csv` | 디지털 헬스케어 | 25,991 |
| `04_바이오헬스_묶음_바이오헬스케어_32454건_…csv` | 두 키워드 묶음 | 32,454 |

### EDA 요약 ([notebooks/eda.ipynb](notebooks/eda.ipynb))
- **묶음 파일 = 두 키워드 파일의 합집합.** 세 파일 합계 65,025행 → 고유 문서 **32,452건** (두 키워드 동시 수집 117건).
- 네이버 URL 형식이 여러 가지라 **언론사ID+기사ID(oid/aid)** 로 문서 ID를 만든다.
- 디지털 헬스케어 : 바이오 의약품 ≈ **4 : 1**, 분기당 1,200~2,300건으로 5년간 공백 없음 (2026Q4는 10/16까지).
- 106개 매체, 상위 10개 매체가 52%.
- 본문 중앙값 약 1,100자. 300자 미만(사진 캡션·속보 한 줄)이 약 590건.
- 약 1만 건 본문 끝에 포털 UI 텍스트(`기사 섹션 분류 안내` 등)가 섞여 있어 제거가 필요하다.
- 하위 주제 커버리지: 바이오 투자 3.6천 · CDMO 2.8천 · 비대면 진료 2.6천 · 바이오시밀러 1.1천 · 의료 AI 규제 850 · **ADC 770 · 기술수출 540 · AI 신약개발 390** → 뒤의 세 주제는 크롤링 보강이 필요하다.
- 뉴스의 **약 22%가 시장 규모·성장률 등 전망 수치**를 인용 → 투자보고서와 상충을 비교할 후보.

## 4. ① 제공 데이터 적재 및 선별 — `collector/load_provided.py`

```
CSV 3종 ─▶ 컬럼 표준화 ─▶ doc_id(oid/aid) 병합 ─▶ 본문 정제 ─▶ 선별 규칙 ─▶ 주제 태깅 ─▶ store/docs.db
                          (수집키워드 합집합)      (보일러플레이트)                       documents / rejected_docs
```

**선별 규칙** (먼저 걸린 규칙 하나를 `drop_reason`으로 기록)

| 사유 | 규칙 | 제외 건수 |
|---|---|---|
| `non_news_section` | 연예·스포츠 섹션 URL | 66 |
| `empty_body` | 본문 없음 | 17 |
| `noise_title` | `[포토]` `[인사]` `[부고]` `[동정]` 등 (`[게시판]` 단신 모음은 유지) | 101 |
| `short_body` | 정제 후 300자 미만 | 544 |
| `off_topic` | 핵심 키워드 0회 + 하위 주제 없음 + 도메인 일반어 3회 미만 | 414 |
| `exact_dup_body` | 본문 완전 중복 (가장 이른 기사만 유지) | 118 |

결과: **고유 32,452건 → 선별 31,192건** (`store/load_provided_stats.json`에 통계 저장)

**`documents` 테이블 주요 컬럼** — 크롤링 문서(특허·보고서)도 같은 스키마로 적재한다.

| 컬럼 | 설명 |
|---|---|
| `doc_id` | `naver:{oid}:{aid}` |
| `source_type` / `evidence_role` | `news` / `current` (특허 `leading`, 보고서 `outlook`) |
| `origin` | `provided` (크롤러는 각자 이름) — 재실행 시 해당 origin만 교체 |
| `title`, `body`, `url`, `published_at`, `publisher` | 정제된 원문·메타데이터 |
| `collect_keywords` | 수집 키워드 (`; ` 구분) |
| `topics` | 하위 주제 태그: ADC · CDMO · AI 신약개발 · 바이오 투자 · 신약 기술수출 · 의료 AI 규제 · 비대면 진료 · 바이오시밀러 |
| `is_outlook` | 시장 전망성 서술 포함 여부 |
| `keyword_hits`, `char_len` | 핵심 키워드 출현 수, 본문 길이 |

근사 중복(통신사 전재 후 일부 수정) 제거, 청킹, 임베딩은 ② 인덱서에서 처리한다.

## 5. ① DART 공시 수집 — `collector/crawl_dart.py`

`.env`에 `DART_API_KEY`가 있어야 한다 (opendart.fss.or.kr 에서 무료 발급).

```
corpCode.xml ─▶ 대상 기업 ─▶ list.json(기업별 공시 목록) ─▶ 공시명 규칙으로 선별 ─▶ document.xml(원문) ─▶ 텍스트 변환 ─▶ documents
              (시드 39곳 / --expand-ksic)                                         (data/dart/raw/ 캐시)
```

**대상 기업**: 하위 주제별 시드 기업 (CDMO·바이오의약품, ADC·플랫폼 기술수출, 제약·신약, AI 신약개발, 디지털 헬스케어·의료 AI).
`--expand-ksic`를 쓰면 상장사 업종코드(21 의약품, 271 의료기기, 70113 의약 R&D)로 대상을 넓힌다. 처음 한 번은 약 4천 번 호출하고, 결과는 캐시한다.

**공시 선별 규칙**

| 분류 (`collect_keywords`) | 공시명 패턴 | 활용 |
|---|---|---|
| 기술이전·라이선스 | 기술도입·이전·수출, 라이선스, 마일스톤·기술료 | 신약 기술수출 1차 근거 |
| 투자판단 주요경영사항 | 투자판단관련주요경영사항 | 임상 승인·결과, 품목허가 |
| 공급계약 | 단일판매ㆍ공급계약체결 | CDMO 수주 |
| 설비투자 | 신규시설투자, 유형자산취득결정 | CDMO 증설 |
| 자금조달 | 유상증자, 전환사채·신주인수권부사채 발행 | 바이오 투자 |
| 타법인 투자·M&A | 타법인주식 취득, 합병, 영업양수 | 사업 확장 |
| 실적전망 (`outlook`) | 영업실적 등에 대한 전망(공정공시) | 기업이 직접 낸 전망 → 뉴스·보고서와 대조 |
| 사업보고서 | 사업보고서 (같은 기간 정정본 중 최신만) | 'II. 사업의 내용'만 추출 |

- 거래소 수시공시(HTML 표)는 `항목 | 값` 형식의 줄로 바꾼다. 제목은 공시 본문의 `제목`/`체결계약명`과 `계약상대`에서 뽑는다.
  - 예: `[삼성바이오로직스] 의약품 위탁생산계약 (계약상대: …)`
- 이미 적재된 접수번호는 건너뛰므로 날마다 다시 실행해도 된다(증분 수집). 요청 한도(020)를 넘으면 그때까지 모은 것만 저장한다.
- 첫 수집 결과 (2021-10-16 ~ 2026-11-04, 시드 39곳): **공시 1,627건**
  - 분류별: 투자판단 623, 공급계약 385, 사업보고서 187, 자금조달 158, 기술이전 105, 타법인 투자·M&A 97, 설비투자 50, 실적전망 22
  - 하위 주제 태그: CDMO 352, 신약 기술수출 281, ADC 112, AI 신약개발 27

```bash
python -m collector.crawl_dart                                   # 시드 기업 전체 (증분)
python -m collector.crawl_dart --companies 알테오젠 리가켐바이오 --since 2024-01-01
python -m collector.crawl_dart --dry-run                         # 선별 결과만 확인
python -m collector.crawl_dart --expand-ksic                     # 업종코드로 대상 확대
```

## 6. ① KIPRIS 특허 수집 — `collector/crawl_kipris.py`

`.env`에 `KIPRIS_API_KEY`가 있어야 한다. 사용하는 KIPRIS Plus 서비스는 아래 5개다.

| 번호 | 서비스 | 용도 |
|---|---|---|
| 1 | 특허·실용 공개·등록공보 | IPC·초록·출원인 검색, 초록, 대표 청구항 |
| 34 | 대표 출원인 | 출원인 표기 정규화 → DART 기업과 연결 (`meta.rpst_applicant`) |
| 27 | 특허 패밀리 | 해외 출원국 (`meta.family_countries`) → 글로벌 진출 선행 신호 |
| 25 | 피인용문헌 | 피인용 수 (`meta.cited_by`) → 핵심 특허 가중치 |
| 48 | 해외특허 | 미국·유럽·PCT 특허 (영문 초록) |

**주제별 검색식** (`TOPIC_QUERIES`, 출원일 2021-10-16 이후)

| 주제 | 검색식 |
|---|---|
| ADC | IPC A61K47/68, 초록 '항체-약물 접합체' |
| AI 신약개발 | IPC G16C20/50, G16C20/70, G16B40, 초록 '인공지능 신약' |
| 의료 AI | IPC G16H50/20(진단 보조), G16H30/40(의료영상) |
| 비대면 진료 | IPC G16H40/67, G16H80 |
| CDMO · 바이오시밀러 | 초록 키워드 |

`--by-applicant`를 쓰면 DART 시드 기업 39곳을 출원인으로 검색한다. 문서 본문은 발명의 명칭·출원인·IPC·요약·대표 청구항으로 구성하고,
`published_at`에는 공개일(근거가 공개된 시점)을, `meta.application_date`에는 출원일(기술 시점)을 둔다. API 응답은 `data/kipris/cache/`에 캐시한다.

**무료 한도(월 1,000회) 운용**: 기본값은 '검색만'이다. 국내 검색 1회 호출로 서지와 초록을 최대 500건 받으므로 주제 전체를 수십 회로 수집할 수 있다.
청구항·대표출원인·패밀리·피인용은 특허마다 1회씩 더 들기 때문에 `--claims`, `--enrich`로 필요할 때만 켠다. `--budget`은 실행 1회의 호출 상한이며, 응답 캐시에서 가져온 것은 세지 않는다.
해외 검색 결과에는 의약·의료 IPC(A61, C07, C12, G16H, G16B)가 붙은 문헌만 남긴다. `G16C20/70` 검색에 수소 저장 같은 무관한 특허가 섞이기 때문이다.

수집 결과 (2026-11-04, 출원일 2021-10-16 이후): **국내 7,872건 + 미국 124건 + 유럽 118건 = 8,114건**
- 국내 주제별(중복 포함): 의료 AI 3,501 · 비대면 진료 1,581 · AI 신약개발 1,509 · ADC 1,310
- 이 중 ADC 161건에만 대표 청구항, 대표출원인, 패밀리, 피인용을 붙였다(패밀리가 있는 것 60건).
- 이번 달 사용량: 약 770회 (검색 66회, 보강 약 650회, 점검·시험 약 50회). 보강 호출이 대부분이라 기본값을 '검색만'으로 바꿨다.

```bash
python -m collector.crawl_kipris --check                          # 서비스별 키 상태
python -m collector.crawl_kipris --foreign US EP --budget 30      # 검색만 (권장)
python -m collector.crawl_kipris --topics ADC --enrich family citing --claims --budget 200
python -m collector.crawl_kipris --budget 0                       # 캐시만으로 재구성 (호출 0회)
```

## 7. ① 산업보고서 수집 — `collector/crawl_reports.py`

BioIN(국가생명공학정책연구센터 포털, bioin.or.kr)의 게시판 4곳에서 **게시물 본문 HTML**만 수집한다. 첨부 PDF는 받지 않는다.

| 게시판 | 내용 | 수집 | 평균 길이 |
|---|---|---|---|
| `watch` BioINdustry | 시장 현황·전망 분석 (요약문 포함 발췌) | 50 | 5.1천 자 |
| `report` BioINpro | 전문가 기술·산업 보고서 | 154 | 5.4천 자 |
| `global` BioINglobal | 해외 동향 보고서 | 27 | 3.7천 자 |
| `industry` 산업 동향 | 유관기관 브리프 재게시 (KOTRA 325, 보건산업진흥원 230, 바이오협회 210, 바이오의약품협회 66, 신약개발재단 59 …) | 1,099 | 2.4천 자 |

- 대상 기간: 2021-10-16 ~ 2026-10-30. 요청 1,421회를 1초 간격으로 순차 실행했다(약 32분). 원문 HTML은 `data/reports/raw/`에 캐시한다.
- `industry` 게시판은 바이오·의료 분류만 받는다. 종합처럼 넓은 분류는 제목에 도메인어가 있을 때만 받고, 농식품·에너지 등은 뺀다. 본문이 300자 미만인 57건은 버렸다.
- `evidence_role` 분포: 시장 규모·성장률 등 전망 서술이 있는 753건은 `outlook`, 나머지 577건은 `current`. 출처 표기(`KOTRA`, `Kotra`, `KOTRA 해외시장뉴스` 등)는 통일했다.
- 사이트가 만든 'AI 요약'은 원문이 아니므로 저장하지 않는다. 원 발행기관과 원문 링크는 `meta.original_publisher`, `meta.original_url`에 남긴다.
- 한국보건산업진흥원 사이트는 브리프 본문을 PDF로만 제공하고, robots.txt가 `/fileDownload`를 금지한다. 그래서 직접 수집하지 않고 BioIN에 본문으로 재게시된 것을 받는다.
- KOTRA 해외시장뉴스(325건)도 이 경로로 확보돼 공공데이터포털 KOTRA API는 신청하지 않았다.

```bash
python -m collector.crawl_reports                                  # 증분 수집 (이미 받은 게시물은 건너뜀)
python -m collector.crawl_reports --boards watch global --max-per-board 20
```

## 8. 크롤링 준수 사항

| 소스 | 수집 방식 | robots.txt / 약관 | 요청 간격 | 출처 URL (`url`) | 수집 일시 |
|---|---|---|---|---|---|
| 제공 뉴스 | 제공 CSV 적재 (직접 크롤링 안 함) | — | — | 네이버 뉴스 원문 URL | `loaded_at` |
| DART | **OpenDART API** (opendart.fss.or.kr) | dart.fss.or.kr은 `/dsaf001/main.do` 등을 수집 금지 → 그 페이지는 요청하지 않고 원문은 API `document.xml`로 받음 | 0.15초, 순차 | 공시 뷰어 링크 (출처 표기용, 직접 요청하지 않음) | `loaded_at` |
| KIPRIS | **KIPRIS Plus API** (plus.kipris.or.kr) | www.kipris.or.kr은 `Disallow: /` → 화면을 크롤링하지 않고 API만 사용 | 0.2초, 순차 | 국내: 공보 DOI `https://doi.org/10.8080/{출원번호}` / 해외: API 요청 URL(키 제외) | `loaded_at` |

| 산업보고서 | BioIN 게시판 HTML 크롤링 | bioin.or.kr은 `/innorix/`, `/InnoAP/`, `/sn3hcv/`, `/sso/`만 금지 → 목록과 본문만 요청, 첨부 PDF는 받지 않음. khidi.or.kr(`/fileDownload` 금지)과 kiet.re.kr(`/common/file/` 금지)은 직접 수집하지 않음 | 1초, 순차 | BioIN 게시물 URL (+ `meta.original_url`) | `loaded_at` |
| 정부 보도자료 | 정책브리핑(korea.kr) 목록·상세 HTML + 보도자료 본문 첨부(HWPX/PDF) 1개 | korea.kr은 `User-Agent: *` 전체 허용. mohw.go.kr은 게시판 대부분 금지 → 직접 수집하지 않음. 별첨·붙임은 받지 않음 | 1초, 순차 | 정책브리핑 보도자료 URL | `loaded_at` |

- 받아 둔 robots.txt는 `data/compliance/`에 날짜와 함께 보관한다.
- 병렬 요청은 쓰지 않는다. 로그인이 필요한 데이터는 수집하지 않는다.
- 개인정보: 발명자명은 받지 않고, 공시책임자·공시담당자 이름과 보도자료 담당자 표(부서·직위·이름·전화)는 저장 전에 지운다 (`storage.scrub_personal`, `crawl_policy.scrub`). 원본 API 응답·HTML 캐시(`data/`, git 제외)는 재현용으로 로컬에만 둔다.
- 모든 문서에 `url`과 `loaded_at`(수집 일시)이 채워져 있는지 확인했다(누락 0건).

## 9. 지식베이스 규모

| 출처 | 원 건수 | 실질 건수 | 예상 청크 (700자 기준) |
|---|---|---|---|
| 제공 뉴스 | 65,025행 (3개 파일) | 31,192 → 3단계 중복 처리 후 **22,623** | 약 46,000 |
| DART 공시 | 1,627 | 1,627 (사업보고서는 'II. 사업의 내용'만 남김) | 약 17,800 |
| KIPRIS 특허 | 8,114 | 8,114 | 약 8,200 |
| BioIN 산업보고서 | 1,330 | 1,330 | 약 6,600 |
| 정부 보도자료 | 1,022 | 1,022 | 약 5,200 |
| **합계** | | **약 34,700** (1차 자료 35%) | **약 83,800** |

- 원 건수와 실질 건수의 차이는 대부분 **파일 간 중복**(묶음 파일 = 키워드 파일의 합집합)이다.
- 같은 보도자료의 재배포본과 같은 사건의 유사 기사는 `collector/dedup_news.py`가 3단계로 처리한다 (31,192 → **22,623**, −27.5%).
  기준·임계값·처리 전후 건수와 설정 근거는 [docs/kb_composition.md](docs/kb_composition.md) §1.
  - 판정 결과: `docs.db`의 `news_dedup` 테이블 (`tier`, `rep_id`, `event_id`), 감사용 목록 `data/compliance/news_dedup_decisions.csv`

**지식베이스 선별 — `collector/select_kb.py`** (적용 완료)

수집한 문서는 모두 `docs.db`에 남기고, 임베딩할 문서에만 `in_kb=1`과 선별 사유(`kb_reason`)를 붙인다. 인덱서는 `WHERE in_kb = 1`인 문서만 읽는다.

| 출처 | 선별 기준 | 지식베이스 | 예상 청크 |
|---|---|---|---|
| 뉴스 | 중복 처리 후 22,623건 → 사건당 최대 3건(21,578) → 점수순으로 분기별 층화 (하위 주제 2 + 전망 서술 1 + 키워드 2회 이상 1 + 적정 길이 0.5) | **8,001** (주제 6,825 · 전망 957 · 키워드 219) | 20,193 |
| DART | 수시공시 전부 + 사업보고서는 2024·2025 사업연도만 ('II. 사업의 내용' 장만) | **1,516** (수시 1,440 · 사업보고서 76) | 9,894 |
| 특허 | 해외 전부, DART 시드 기업 출원 전부, 국내는 주제별 할당 (ADC·AI 신약 전부, 의료 AI 600, 비대면 진료 350, 보강·등록·최근 순) | **4,109** | 4,196 |
| 산업보고서 | 분석 보고서(watch·report·global) 231건 전부 + 산업 동향 브리프는 하위 주제·전망 점수순 500건 | **731** | 3,619 |
| 정부 보도자료 | 관련성 필터를 통과한 전부 | **1,022** | 5,173 |
| **합계** | | **15,379 (1차 자료 48%)** | **약 43,100 (1차 자료 53%)** |

- 연도별 뉴스: 2021(10월~) 287 · 2022 1,475 · 2023 1,622 · 2024 1,403 · 2025 1,846 · 2026(~10월) 1,364
- 크롤러가 문서를 다시 적재하면 `in_kb`가 0으로 돌아가므로, **수집 후에는 항상 `select_kb`를 다시 실행**한다(1분 이내. 뉴스 중복 판정은 `dedup_news`가 미리 계산).
- 일정이 밀리면 `--news-target`(기본 8,000)부터 줄인다.

```bash
python -m collector.select_kb
python -m collector.select_kb --news-target 6000 --dry-run    # 축소 시 구성 미리보기
```

## 10. 실행 순서

```bash
pip install -r requirements.txt
```

```bash
# 제공 CSV를 data/provided/ 에 둔 뒤 (다른 위치면 --src 로 지정)
python -m collector.load_provided
```

```bash
# DB에 쓰지 않고 선별 통계만 확인
python -m collector.load_provided --dry-run
```

```bash
# DART 공시 수집 (.env 에 DART_API_KEY 필요)
python -m collector.crawl_dart
```

```bash
# KIPRIS 특허 수집 — 검색만, 실행당 호출 상한 (.env 에 KIPRIS_API_KEY 필요)
python -m collector.crawl_kipris --foreign US EP --budget 30
```

```bash
# 산업보고서 크롤링 (BioIN, 1초 간격 · 첫 실행 약 30분)
python -m collector.crawl_reports
```

```bash
# 정부 보도자료 (정책브리핑, 1초 간격 · 첫 실행 약 1시간)
python -m collector.crawl_policy
```

```bash
# 뉴스 중복 처리 → 지식베이스 선별 (수집 후 항상 이 순서로 마지막에)
python -m collector.dedup_news
```

```bash
python -m collector.select_kb
```

```bash
# ② 인덱싱: 청킹(설정 3종) → 임베딩 (.env 에 OPENAI_API_KEY 필요, 설정당 약 25분·$0.5)
python -m indexer chunk
```

```bash
python -m indexer embed --configs structure
```

```bash
# 문서 수·청크 수·임베딩 수 대조 (chunk·embed 끝에 자동 실행, 실패 시 종료 코드 1)
python -m indexer verify
```

```bash
# 벡터 DB 적재 (structure → store/chroma/, 캐시 벡터만 사용 · 약 2분) + 적재 검증
python -m indexer load
```

```bash
# 적재 검증만 다시 (개수·빈 청크·벡터·영속화·샘플 검색 → store/vector_load_stats.json)
python -m indexer verify-load
```

```bash
# 하이브리드 검색용 BM25 인덱스 (structure → store/bm25/structure.npz, 약 30초)
python -m indexer bm25
```

```bash
# 평가: 대표 질문 근거 커버리지, 청킹 설정별 Hit@k·MRR
python -m eval.kb_coverage
```

```bash
python -m eval.retrieval_eval
```

```bash
# 평가 하네스: 챗봇과 같은 체인으로 79문항 실행 + LLM 채점 (약 5분) → eval/results/answer_eval_*.json, answer_review_*.csv
python -m eval.run_eval
```

```bash
# 일부 문항만, 채점 생략
python -m eval.run_eval --ids F01 U01 --no-judge
```

```bash
# 청킹 설정 비교 (먼저 python -m indexer load --configs fixed structure_notitle)
python -m eval.run_eval --chunk-config fixed
```

```bash
# 검색 방식 비교: dense | hybrid | hybrid_time(기본) | hybrid_rerank(+LLM 재정렬)
python -m eval.run_eval --retrieval hybrid_rerank
```

```bash
# ④ 챗봇 UI — store/chroma/ 와 store/docs.db 만 있으면 인덱서 없이 구동 (http://localhost:8501)
streamlit run app/main.py
```

옵션: `--src <파일|디렉터리>`, `--db store/docs.db`, `--since 2023-01-01`, `--until 2026-06-30`

EDA 노트북은 `notebooks/eda.ipynb`를 열어 위에서부터 실행한다 (한글 폰트: Windows `Malgun Gothic`).

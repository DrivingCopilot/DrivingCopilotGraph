# DrivingCopilotGraph — 차량 지식 적재 계층 (Graph + Vector RAG)

**On-Device Multimodal Driving Copilot** 프로젝트의 **지식 적재(ingestion)·스키마 권위** 레포입니다.
차량 매뉴얼(PDF)과 진단 텍스트를 받아, 하류 에이전트/백엔드가 검색에 사용할 **두 개의 지식 저장소**를 구축합니다.

- **Graph RAG (Neo4j)** — 로컬 LLM(`Qwen3-VL-4B-Instruct`, HuggingFace `transformers`)으로 텍스트에서
  엔티티/관계를 추출해 Neo4j 지식 그래프로 적재합니다. 그래프 스키마(노드/관계 정의)의 단일 권위가 이 레포입니다.
- **Vector RAG (Qdrant)** — PDF를 파싱·시맨틱 청킹(`bge-m3`)한 뒤 임베딩해 Qdrant 컬렉션으로 인덱싱합니다.

두 파이프라인은 **동일한 PDF 파서 + 시맨틱 청커**를 공유합니다. 따라서 Qdrant에 들어가는 청크와
Neo4j 그래프로 추출되는 청크가 **같은 원본 텍스트**라서, 벡터/그래프 검색이 서로 어긋나지 않습니다.

> **이 레포의 역할은 "적재"입니다** — 질의 응답 서빙(API), 멀티에이전트 오케스트레이션은 형제 레포가 담당합니다.
> 이 레포는 그 레포들이 읽어갈 Neo4j 그래프와 Qdrant 인덱스를 만들어 두는 계층입니다.

## 멀티레포 생태계에서의 위치

| 레포 | 역할 | 이 레포와의 관계 |
|------|------|------------------|
| **DrivingCopilotGraph** (이 레포) | 지식 적재 · 그래프 스키마 권위 | Neo4j 그래프 · Qdrant 인덱스를 **생성** |
| DrivingCopilotBackend | 서빙(FastAPI) · Qdrant 서버 호스팅 | Qdrant(6333)를 `docker-compose`로 **띄우는 쪽**. 이 레포의 적재가 그 서버에 write |
| Agent | A2A 멀티에이전트 오케스트레이션 | 적재된 그래프/벡터를 **검색**해 사용 |

> Neo4j 접속 키는 Backend와 동일(`NEO4J_USER`), Qdrant 컬렉션명/URL 기본값도 Backend와 일치시켜
> 같은 저장소를 공유하도록 맞춰져 있습니다. (Agent 레포는 `NEO4J_USERNAME`을 써서 키가 다름 — 주의)

## 데이터 흐름

```
                              ┌─▶ 임베딩(bge-m3) ──────────▶ Qdrant  (Vector RAG)
PDF ─▶ VehiclePDFParser ─▶ SemanticChunker ─(같은 청크)─┤
  또는 .txt(문단 분할)                                    └─▶ 로컬 LLM 엔티티/관계 추출 ─▶ Neo4j (Graph RAG)
                                                              (Qwen3-VL-4B-Instruct)
```

## 레포지토리 구조

| 경로 | 설명 |
|------|------|
| `core/config.py` | 전역 설정. Neo4j/Qdrant 접속, 로컬 LLM(모델·device·토큰) 값을 env 우선 로드 |
| `graph/schema.py` | **그래프 스키마 권위** — 노드 8종 / 관계 10종 정의 (`DrivingGraphSchema`) |
| `graph/graph_rag.py` | `LocalQwen3VL`(HF LLM 래퍼) + `VehicleGraphManager`(추출·적재·1~2 hop 탐색·컨텍스트 융합) |
| `graph/ingest.py` | **Graph RAG 적재 엔트리포인트** (`python -m graph.ingest`) — PDF/txt/Qdrant → Neo4j |
| `graph/vector_rag.py`, `graph/vector_services.py` | 그래프 레포 내부에서 벡터 파이프라인을 자체 실행할 때 쓰는 오케스트레이터(파서/청커/임베더 번들) |
| `services/pdf_parser.py` | `VehiclePDFParser` — PyMuPDF 기반 PDF → `Document` 파싱(짧은 노이즈 페이지 제거) |
| `services/semantic_chunker.py` | `SemanticChunker` — bge-m3 임베딩 기반 시맨틱 경계 청킹 |
| `services/embedder.py` | `VehicleEmbedder` — 청크 임베딩 + Qdrant upsert/search |
| `services/index_manuals.py` | **Vector RAG 적재 엔트리포인트** (`python -m services.index_manuals`) — PDF → Qdrant |
| `scripts/01~03_*.sh` | Neo4j 기동 → 로컬 LLM 환경 구성 → LLM 적재 순차 스크립트 |
| `scripts/test/` | LLM 없이 Cypher 시드로 그래프 스키마/탐색을 검증하는 테스트 |
| `docker-compose.yml` | Neo4j 5.26 서버 정의(포트 `7474`/`7687`, apoc, `./neo4j` 볼륨). **Qdrant는 미포함**(Backend 소관) |
| `requirements-docker.txt` | 적재 런타임 파이프라인 의존성의 **단일 소스**(그래프+벡터 전체 스택). `02_setup_llm.sh`·Docker가 사용 |
| `requirements.txt` | 프로젝트 전체(서버/평가 예정 포함)의 폭넓은 목록. 런타임 파이프라인은 위 파일 참조 |

## 사전 요구사항

- **Docker / Docker Compose** — Neo4j 컨테이너 기동용
- **Python 3.10+** — 적재 파이프라인 및 로컬 LLM 실행용 (`scripts/02_setup_llm.sh`가 `.venv` 생성)
- **CUDA GPU (적재 시 권장)** — 로컬 LLM(`Qwen3-VL-4B-Instruct`) 추론용. `GRAPH_LLM_DEVICE=auto`(기본)면
  GPU가 없을 때 자동으로 CPU로 폴백하지만, CPU 추론은 매우 느립니다. **Neo4j 서버 기동/Cypher 테스트에는 GPU가 불필요**합니다.

## 설정 파일

| 파일 | 설명 |
|------|------|
| `.env.example` | 환경 변수 템플릿. 복사해 `.env`로 사용 (`NEO4J_*`, `GRAPH_MAX_RESULTS`, `QDRANT_*`) |
| `.env` | 실제 접속값. **커밋 금지**(`.gitignore` 처리). 없으면 `01_start_neo4j.sh`가 자동 생성 |
| `core/config.py` | 모든 상수를 env 우선으로 로드(기본값 fallback). 아래 표의 값들이 여기서 통합 관리됨 |

주요 환경 변수(모두 `.env` 또는 셸 env로 덮어쓰기 가능):

| 변수 | 기본값 | 용도 |
|------|--------|------|
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` / `NEO4J_DATABASE` | `bolt://localhost:7687` / `neo4j` / `password` / `neo4j` | Neo4j 접속 |
| `GRAPH_MAX_RESULTS` | `20` | 그래프 탐색 결과 상한 |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant **서버 모드** 접속 대상(Backend가 호스팅) |
| `QDRANT_PATH` | (비어 있음) | 지정 시 로컬 **파일 모드**(별도 서버 불필요) |
| `COLLECTION_NAME` | `vehicle_manuals` | Qdrant 컬렉션명(Backend와 동일) |
| `GRAPH_LLM_MODEL` | `Qwen/Qwen3-VL-4B-Instruct` | 추출/융합용 로컬 LLM HuggingFace repo id |
| `GRAPH_LLM_DEVICE` | `auto` | `auto`=GPU 있으면 cuda·없으면 cpu / `cuda` / `cpu` 명시 가능 |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace 가중치 캐시 루트(Docker 볼륨 마운트 지점) |

기본 접속값은 `neo4j / password`이며, 비밀번호를 바꾸려면 `.env`의 `NEO4J_PASSWORD`만 수정하면
`docker-compose.yml`과 `core/config.py`가 함께 참조합니다.

## 빠른 시작 (Graph RAG)

Neo4j 지식 그래프를 서버 기동 → 스키마 검증 → LLM 적재 순서로 구축합니다.

```bash
# 1) Neo4j 서버 기동 (.env 자동 생성 → 컨테이너 up → healthcheck 대기 → 접속 검증)
./scripts/01_start_neo4j.sh

# 2) LLM 없이 그래프 스키마/탐색 검증 (Cypher 시드 직접 적재)
./scripts/test/run_cypher_test.sh          # 노드/관계 수 + 1~2 hop 탐색 결과 출력
./scripts/test/run_cypher_test.sh --clean  # 테스트 데이터(_test=true) 정리

# 3) 로컬 LLM 환경 구성 (GPU 환경 권장)
./scripts/02_setup_llm.sh
#    - .venv 생성 + requirements-docker.txt 설치 + Qwen3-VL-4B-Instruct 가중치 다운로드
#    - 모델 변경   : GRAPH_LLM_MODEL=<HF repo id> ./scripts/02_setup_llm.sh
#    - 캐시 위치   : HF_HOME=/data/hf ./scripts/02_setup_llm.sh   (Docker 볼륨 마운트 지점)
#    - 의존성만    : SKIP_MODEL_DOWNLOAD=1 ./scripts/02_setup_llm.sh

# 4) LLM 기반 적재 — 텍스트에서 엔티티/관계 추출 후 그래프 적재
./scripts/03_ingest.sh <입력파일_또는_디렉터리>
```

> GPU가 없는 환경에서는 3~4단계를 건너뛰고 **1~2단계(Cypher 테스트)** 만으로 스키마/탐색을 검증할 수 있습니다.

### 적재 엔트리포인트 직접 실행 (`graph.ingest`)

`03_ingest.sh`는 아래 엔트리포인트를 감싼 것으로, `.venv` 활성화 후 직접 호출하면 더 많은 옵션을 쓸 수 있습니다.

```bash
source .venv/bin/activate

python -m graph.ingest data/manuals/            # 디렉터리 내 모든 pdf/txt
python -m graph.ingest manual.pdf --max-chunks 20   # 앞 20개 청크만(스모크 테스트)
python -m graph.ingest notes.txt                # .txt 는 빈 줄(문단) 분할 → 임베딩 스택 불필요
python -m graph.ingest manual.pdf --concurrency 3   # 청크 병렬 추출(쓰기는 직렬화). 8GB VRAM 은 2~3 권장
python -m graph.ingest --qdrant --qdrant-path services/qdrant_storage
#   ▲ 이미 Qdrant에 인덱싱된 청크를 그대로 재사용해 그래프로 적재(벡터/그래프 단일 원본 공유, 원본 PDF·torch 불필요)
```

- **PDF 입력**은 시맨틱 청킹을 위해 벡터 스택(`pymupdf`, `langchain-*`, `sentence-transformers`)이 필요합니다.
- **`.txt` 입력**은 문단 분할만 하므로 임베딩 스택 없이도 그래프 추출을 시험할 수 있습니다.

## 빠른 시작 (Vector RAG — Qdrant 인덱싱)

PDF를 파싱·청킹·임베딩해 Qdrant 컬렉션으로 인덱싱합니다. 접속 모드는 `QDRANT_PATH` 유무로 결정됩니다.

| `QDRANT_PATH` | 모드 | 접속 대상 |
|---|---|---|
| 비어 있음(기본) | **서버 모드** | `QDRANT_URL` (기본 `http://localhost:6333`) |
| 경로 지정 | 로컬 파일 모드 | 해당 디렉터리 (별도 서버 불필요) |

```bash
source .venv/bin/activate

# 청크 결과만 확인 (Qdrant 불필요)
python -m services.index_manuals manuals/매뉴얼.pdf --step chunk

# 전체 파이프라인 (파싱 + 청킹 + 임베딩 + Qdrant 저장)
python -m services.index_manuals manuals/매뉴얼.pdf --step embed
```

> **중요:** 서버 모드가 접속하는 Qdrant 서버(`6333`)는 **`DrivingCopilotBackend` 레포의
> `docker-compose.yml`이 호스팅**하며 **이 레포에는 포함되어 있지 않습니다**(이 레포의
> `docker-compose.yml`은 Neo4j 전용). 기본값 그대로 실행하려면 Backend의 Qdrant 컨테이너가 먼저 떠 있어야 합니다.
>
> Backend 없이 단독으로 시험하려면 로컬 파일 모드로 지정하세요:
>
> ```bash
> QDRANT_PATH=./qdrant_storage python -m services.index_manuals manuals/매뉴얼.pdf --step embed
> ```
>
> `COLLECTION_NAME`/`QDRANT_URL` 기본값(`vehicle_manuals` / `localhost:6333`)은 Backend
> `app/config.py`와 동일해, 서버 모드에서 같은 컬렉션을 공유합니다.

## 그래프 스키마

`graph/schema.py`의 `DrivingGraphSchema`가 추출·적재의 스키마 권위입니다. 노드마다 **식별 속성이 다르다**는 점이
탐색(`retrieve_context`)의 핵심이라, 매칭 시 `name`/`code`/`type`/`description`/`id`를 모두 후보로 봅니다.

- **노드(8종)**: `Component`(name), `WarningLight`(name), `Symptom`(description), `Maintenance`(type),
  `DTC Code`(code), `System`(name), `Action`(description), `Schedule`(value/unit)
- **관계(10종)**: `HAS_PART`, `MAINTAINED_BY`, `INDICATES`, `CAUSED_BY`, `SYMPTOM_OF`, `RESOLVED_BY`,
  `APPLIES_TO`, `HAS_INTERVAL`, `TRIGGERS`, `MAPS_TO`

## 접속 정보

| 항목 | 값 |
|------|-----|
| Neo4j Browser | http://localhost:7474 |
| Bolt (드라이버) | `bolt://localhost:7687` |
| 계정 | `neo4j` / `password` (기본값) |
| 데이터 볼륨 | `./neo4j/data` (컨테이너 재시작해도 유지) |

서버 중지는 `docker compose down`이며, 그래프 데이터는 볼륨에 보존됩니다.

## 추가 예정 사항

- **Docker 이미지 패키징** — 적재 파이프라인(그래프+벡터)을 GPU 배치 이미지로 패키징(멀티스테이지 CUDA +
  `HF_HOME` 볼륨 마운트 + compose `ingest` 프로파일). 선행 정리(의존성 단일화·device 자동 폴백·HF 캐시 문서화)는 완료.
- **Knowledge Agent MCP Tools** — `VehicleGraphManager`의 추출/탐색 기능을 MCP(Model Context Protocol)
  도구로 노출하는 전용 폴더가 향후 추가될 예정입니다(현재 `mcp_run_extraction` 래퍼가 그 기반).

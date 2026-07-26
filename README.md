# On-Device Multimodal Driving Copilot

**On-Device Multimodal Driving Copilot** 레포지토리에 오신 것을 환영합니다! 이 프로젝트는 LangChain과 LangGraph를 활용하여 복잡한 워크플로우, 추론 및 도구(Tool) 실행을 관리하는 고급 멀티모달 차량용 AI 비서를 구현합니다.

## 레포지토리 구조

- **`graph/`**: 핵심 상태 그래프(StateGraph) 정의 및 RAG 구현체를 포함합니다.
  - `graph_rag.py`
  - `vector_rag.py`
  - *(예정)* Text2SQL 에이전트 구현체.
- **`services/`**: 시맨틱 청커(Semantic Chunker), 임베더(Embedder), PDF 파서(Parser)와 같은 서비스 모듈 및 데이터 처리 스크립트를 보관합니다.

## Graph RAG (Neo4j) 실행 가이드

Graph RAG는 차량 진단 지식을 Neo4j 지식 그래프로 관리합니다. 아래 순서대로 실행하면
서버 기동부터 테스트 데이터 검증, LLM 기반 적재까지 진행할 수 있습니다.

### 사전 요구사항

- **Docker / Docker Compose** — Neo4j 컨테이너 기동용
- **Python 3.10+** — 적재 파이프라인 및 LLM 실행용 (`scripts/02_setup_llm.sh`에서 venv 생성)
- **(적재 시) CUDA GPU 환경** — 로컬 LLM(`Qwen3-VL-4B-Instruct`) 추론용. 서버 기동/테스트에는 불필요합니다.

### 설정 파일

| 파일 | 설명 |
|------|------|
| `docker-compose.yml` | Neo4j 5.26 서버 정의 (포트 `7474`/`7687`, apoc 플러그인, `./neo4j` 볼륨) |
| `.env.example` | 환경 변수 템플릿. 복사해 `.env`로 사용 (`NEO4J_*`, `GRAPH_MAX_RESULTS`) |
| `.env` | 실제 접속값. **커밋 금지**(`.gitignore` 처리됨). 없으면 `01_start_neo4j.sh`가 자동 생성 |
| `core/config.py` | `NEO4J_URI/USER/PASSWORD/DATABASE` 등을 env 우선으로 로드 (기본값 fallback) |

기본 접속값은 `neo4j / password`이며, 비밀번호를 바꾸려면 `.env`의 `NEO4J_PASSWORD`만 수정하면
`docker-compose.yml`과 `core/config.py`가 함께 참조합니다.

### 실행 순서

```bash
# 1) Neo4j 서버 기동 (.env 자동 생성 → 컨테이너 up → healthcheck 대기 → 접속 검증)
./scripts/01_start_neo4j.sh

# 2) 테스트 데이터로 그래프 동작 검증 (LLM 없이 Cypher 직접 적재)
./scripts/test/run_cypher_test.sh
#    - 노드/관계 수와 1~2 hop 탐색 결과를 출력
#    - 정리:  ./scripts/test/run_cypher_test.sh --clean

# 3) (선택) 로컬 LLM 환경 구성 — GPU 환경에서만
./scripts/02_setup_llm.sh
#    - .venv 생성 + 의존성 설치 + Qwen3-VL-4B-Instruct 다운로드(HuggingFace)
#    - 모델 변경: GRAPH_LLM_MODEL=<HF repo id> ./scripts/02_setup_llm.sh

# 4) LLM 기반 적재 — 텍스트에서 엔티티/관계 추출 후 그래프 적재
./scripts/03_ingest.sh <입력파일_또는_디렉터리>
```

### 접속 정보

| 항목 | 값 |
|------|-----|
| Neo4j Browser | http://localhost:7474 |
| Bolt (드라이버) | `bolt://localhost:7687` |
| 계정 | `neo4j` / `password` (기본값) |
| 데이터 볼륨 | `./neo4j/data` (컨테이너 재시작해도 유지) |

서버 중지는 `docker compose down`이며, 그래프 데이터는 볼륨에 보존됩니다.

> **참고:** `03_ingest.sh`는 `graph/ingest.py` 엔트리포인트를 통해
> `graph/graph_rag.py`의 `LocalQwen3VL`(HuggingFace `transformers` 기반) 로
> 엔티티/관계를 추출한 뒤 Neo4j 에 적재합니다. GPU 미보유 환경에서는
> `scripts/02_setup_llm.sh` 단계를 건너뛰고 `run_cypher_test.sh` 로만 검증하세요.

## 추가 예정 사항

**Knowledge Agent MCP Tools**: Knowledge Agent를 위한 포괄적인 MCP(Model Context Protocol) 도구들을 관리하고 보관할 전용 폴더가 향후 생성될 예정입니다.
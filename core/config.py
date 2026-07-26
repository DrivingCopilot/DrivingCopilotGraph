# core/config.py
#
# 프로젝트 전역 설정 상수 관리 모듈.
# 모델, Qdrant, 경로 등 여러 서비스에서 공유하는 설정값을 한 곳에서 관리한다.
# 설정 변경 시 이 파일만 수정하면 된다.

import os

# ---------------------------------------------------------------------------
# 임베딩 모델
# ---------------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-m3"   # 임베딩 모델. A6000 통합 시 vLLM으로 교체
VECTOR_SIZE = 1024             # bge-m3 dense 벡터 차원

# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

# 접속 설정은 Backend(app/config.py)의 Qdrant 설정과 동일하게 맞춘다.
# Backend 가 docker-compose 로 Qdrant 를 서버 모드(6333)로 띄우고, 이 레포의 적재는
# 그 서버에 write 한다. QDRANT_PATH 가 설정되면 로컬 파일 모드, 비어 있으면 서버 모드(권장).
QDRANT_PATH = os.getenv("QDRANT_PATH", "")                       # 비우면 서버 모드
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")    # Backend qdrant 서비스
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "vehicle_manuals")

# 아래 파서/청커 값은 정의만 되어 있고, 현재 VehiclePDFParser/SemanticChunker 는
# 이 상수를 읽지 않고 코드 내 하드코딩 기본값(각각 20자 / 64토큰 / standard_deviation)을 쓴다.
# ---------------------------------------------------------------------------
# 파서
# ---------------------------------------------------------------------------

MIN_TEXT_LENGTH = 20   # 이 길이 미만 페이지는 노이즈로 제거 (VehiclePDFParser)

# ---------------------------------------------------------------------------
# 청커
# ---------------------------------------------------------------------------

CHUNK_MIN_TOKENS = 64                          # 이 토큰 수 미만 청크는 앞 청크에 병합
BREAKPOINT_THRESHOLD_TYPE = "standard_deviation"  # 시맨틱 경계 감지 방식

# ---------------------------------------------------------------------------
# 서버 (Agent supervisor / Backend)
# ---------------------------------------------------------------------------

AGENT_HOST = os.getenv("AGENT_HOST", "0.0.0.0")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8001"))                # supervisor FastAPI 포트
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")  # DrivingCopilotBackend
ALLOWED_ORIGINS = [
    "http://localhost:3000",   # React frontend
    "http://localhost:8000",   # FastAPI backend (server-to-server 호출용)
]

# ---------------------------------------------------------------------------
# Neo4j (Graph RAG)
# ---------------------------------------------------------------------------
# DrivingCopilotBackend app/config.py 와 동일한 키를 사용한다.
# (주의: Agent 레포는 NEO4J_USERNAME 을 쓰지만, 여기·Backend는 NEO4J_USER 로 통일)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")   # 로컬 개발 기본값
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GRAPH_MAX_RESULTS = int(os.getenv("GRAPH_MAX_RESULTS", "20"))  # 그래프 탐색 결과 상한

# ---------------------------------------------------------------------------
# 로컬 LLM (HuggingFace / Qwen3-VL)
# ---------------------------------------------------------------------------
# Graph RAG 엔티티/관계 추출 및 컨텍스트 융합에 사용하는 로컬 VLM.
# scripts/02_setup_llm.sh 로 HuggingFace Hub 에서 가중치를 사전 다운로드한다.
# 적재(엔티티/관계 추출)는 텍스트만 다루므로 가벼운 4B Instruct 로 충분하다.
GRAPH_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
# "auto" = GPU 있으면 cuda, 없으면 cpu 로 자동 선택(비-GPU 환경/CI 에서도 로딩이 죽지 않음).
# "cuda"/"cpu" 등 명시값은 그대로 transformers device_map 으로 전달된다.
GRAPH_LLM_DEVICE = os.getenv("GRAPH_LLM_DEVICE", "auto")  # transformers device_map
GRAPH_LLM_MAX_NEW_TOKENS = int(os.getenv("GRAPH_LLM_MAX_NEW_TOKENS", "1024"))
GRAPH_LLM_TEMPERATURE = float(os.getenv("GRAPH_LLM_TEMPERATURE", "0.0"))  # 추출은 결정적으로

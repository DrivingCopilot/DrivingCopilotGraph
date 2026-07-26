#!/usr/bin/env bash
# 03_ingest.sh
# -----------------------------------------------------------------------------
# 로컬 LLM(HuggingFace transformers / Qwen3-VL-4B-Instruct)을 사용해 차량 매뉴얼(PDF)·
# 진단 텍스트(.txt)에서 엔티티/관계를 추출하고 Neo4j 지식 그래프에 적재한다.
# (graph/ingest.py 엔트리포인트)
#
# PDF 는 Vector(Qdrant) 파이프라인과 동일한 VehiclePDFParser + SemanticChunker 로
# 청킹되어, 벡터/그래프가 같은 원본 청크를 공유한다.
#
# 전제:
#   - ./scripts/01_start_neo4j.sh 로 Neo4j 가 기동되어 있을 것
#   - ./scripts/02_setup_llm.sh 로 .venv / transformers / Qwen3-VL-4B-Instruct 가중치 준비될 것
#   - PDF 입력 시: 벡터 청킹 스택(pymupdf, langchain-*, sentence-transformers) 설치 필요
#                 (.txt 입력은 추가 의존성 없이 동작)
#
# 사용법:
#   ./scripts/03_ingest.sh <입력파일 또는 디렉터리>
#
# 환경 변수(선택):
#   VENV_DIR   가상환경 경로 (기본 .venv)
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
INPUT="${1:-}"

if [[ -z "$INPUT" ]]; then
  echo "사용법: $0 <입력파일 또는 디렉터리>" >&2
  exit 2
fi
if [[ ! -e "$INPUT" ]]; then
  echo "!! 입력 경로가 존재하지 않습니다: $INPUT" >&2
  exit 2
fi

# --- .env / venv 준비 ------------------------------------------------------
if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  source .env; set +a
fi
if [[ ! -d "$VENV_DIR" ]]; then
  echo "!! 가상환경($VENV_DIR)이 없습니다. 먼저 ./scripts/02_setup_llm.sh 를 실행하세요." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- Neo4j 연결 확인 -------------------------------------------------------
CONTAINER="driving-copilot-neo4j"
if ! docker exec "$CONTAINER" cypher-shell -u "${NEO4J_USER:-neo4j}" -p "${NEO4J_PASSWORD:-password}" \
      "RETURN 1;" >/dev/null 2>&1; then
  echo "!! Neo4j 에 접속할 수 없습니다. 먼저 ./scripts/01_start_neo4j.sh 를 실행하세요." >&2
  exit 1
fi

# --- 적재 실행 -------------------------------------------------------------
# graph.ingest 는 VehicleGraphManager.extracting_data 를 호출해 추출→적재하는 엔트리포인트.
echo "▶ LLM 기반 적재 시작: $INPUT"
python -m graph.ingest "$INPUT"

echo "✅ 적재 완료. 확인: http://localhost:7474"

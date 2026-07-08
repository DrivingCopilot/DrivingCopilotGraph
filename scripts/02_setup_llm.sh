#!/usr/bin/env bash
# 02_setup_llm.sh
# -----------------------------------------------------------------------------
# Graph RAG 엔티티/관계 추출에 사용할 로컬 LLM(Qwen2-VL-7B INT4/AWQ) 환경을 구성한다.
#
# 대상: GPU 워크스테이션(별도 로컬 환경). 이 스크립트는 특정 머신에 종속되지 않으며,
#       CUDA 가 준비된 환경에서 실행하는 것을 전제로 한다.
#
# 수행 내용:
#   1) Python 가상환경(.venv) 생성
#   2) 추론/그래프 의존성 설치 (torch, transformers, autoawq, neo4j-graphrag 등)
#   3) LLM 가중치 다운로드 (기본: Qwen/Qwen2-VL-7B-Instruct-AWQ)
#
# 환경 변수(선택):
#   LLM_MODEL   내려받을 HF 모델 ID (기본 Qwen/Qwen2-VL-7B-Instruct-AWQ)
#   VENV_DIR    가상환경 경로       (기본 .venv)
#   SKIP_MODEL_DOWNLOAD=1  가중치 다운로드 건너뛰기(의존성만 설치)
#
# 사용법:  ./scripts/02_setup_llm.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2-VL-7B-Instruct-AWQ}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- 0. 사전 점검 ----------------------------------------------------------
echo "[0/3] 환경 점검"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "!! $PYTHON_BIN 을 찾을 수 없습니다. Python 3.10+ 를 설치하세요." >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo "   경고: nvidia-smi 미검출. GPU/CUDA 환경에서 실행하는지 확인하세요."
fi

# --- 1. 가상환경 ----------------------------------------------------------
echo "[1/3] 가상환경 준비: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- 2. 의존성 설치 -------------------------------------------------------
echo "[2/3] 의존성 설치"
# 프로젝트 공통 의존성
python -m pip install -r requirements.txt

# LLM(Qwen2-VL AWQ) 추론 + Graph RAG 파이프라인 의존성
#   - torch          : PyTorch (CUDA 휠은 실행 환경의 CUDA 버전에 맞게 자동 선택)
#   - transformers   : Qwen2-VL 지원 (>=4.45)
#   - accelerate     : 디바이스 매핑/오프로드
#   - autoawq        : AWQ(INT4) 양자화 가중치 로드
#   - qwen-vl-utils  : Qwen2-VL 전처리 유틸
#   - neo4j-graphrag : LLMEntityRelationExtractor / SchemaBuilder (graph/ 에서 사용)
#   - huggingface_hub: 모델 다운로드 CLI
python -m pip install \
  "torch" \
  "transformers>=4.45.0" \
  "accelerate>=0.34.0" \
  "autoawq" \
  "qwen-vl-utils" \
  "neo4j-graphrag" \
  "huggingface_hub[cli]"

# --- 3. 모델 다운로드 -----------------------------------------------------
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "[3/3] SKIP_MODEL_DOWNLOAD=1 → 가중치 다운로드 생략"
else
  echo "[3/3] LLM 가중치 다운로드: $LLM_MODEL"
  # 게이트/사설 모델이면 먼저 'huggingface-cli login' 필요
  huggingface-cli download "$LLM_MODEL" --local-dir "models/$(basename "$LLM_MODEL")"
fi

cat <<EOF

✅ LLM 환경 구성 완료
   - venv    : $VENV_DIR   (활성화: source $VENV_DIR/bin/activate)
   - model   : $LLM_MODEL  → models/$(basename "$LLM_MODEL")

다음 단계:
   1) graph/graph_rag.py 의 LocalQwen2VL._agenerate 에 실제 추론 바인딩 구현
      (models/$(basename "$LLM_MODEL") 로드 → generate)
   2) 적재 실행: ./scripts/03_ingest.sh <입력.txt 또는 디렉터리>
EOF

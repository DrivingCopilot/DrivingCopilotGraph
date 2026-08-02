#!/usr/bin/env bash
# 02_setup_llm.sh
# -----------------------------------------------------------------------------
# Graph RAG 엔티티/관계 추출에 사용할 로컬 VLM(Qwen3-VL-4B-Instruct)을
# HuggingFace 에서 직접 내려받아 transformers 로 서빙할 환경을 구성한다.
#
# 구성 원칙:
#   - 모델 가중치       : HuggingFace Hub 에서 huggingface_hub 로 직접 다운로드(사전 캐싱)
#   - Python 의존성     : 반드시 프로젝트 가상환경(.venv) 안에서만 설치
#
# 대상: CUDA GPU 워크스테이션. 적재(엔티티/관계 추출)는 텍스트만 다루므로
#       가벼운 4B Instruct 로 충분하다(비전 인코더는 로드되지만 사용하지 않음).
#
# 수행 내용:
#   1) Python 가상환경(.venv) 생성 + requirements-docker.txt 로 파이프라인 의존성 설치
#      (Docker 이미지와 동일한 목록 → venv/컨테이너 환경 일치)
#   2) HuggingFace Hub 에서 GRAPH_LLM_MODEL 가중치 사전 다운로드(캐시)
#
# 환경 변수(선택):
#   GRAPH_LLM_MODEL   내려받을 HuggingFace repo id (기본 Qwen/Qwen3-VL-4B-Instruct)
#   VENV_DIR          가상환경 경로              (기본 .venv)
#   HF_HOME           HuggingFace 캐시 루트(가중치 저장 위치). Docker 에서는 이 경로를
#                     볼륨으로 마운트해 이미지에 가중치를 굽지 않는다. 기본 ~/.cache/huggingface
#   HF_TOKEN          비공개/게이트 모델 접근용 HuggingFace 토큰(선택)
#   SKIP_MODEL_DOWNLOAD=1  가중치 다운로드 생략(의존성만 구성)
#
# 사용법:  ./scripts/02_setup_llm.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 구조화 추출 파이프라인(core/config.py GRAPH_LLM_MODEL)과 반드시 동일한 repo id 를 받아야 한다.
GRAPH_LLM_MODEL="${GRAPH_LLM_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# --- 0. 사전 점검 ----------------------------------------------------------
echo "[0/2] 환경 점검"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "!! $PYTHON_BIN 을 찾을 수 없습니다. Python 3.10+ 를 설치하세요." >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo "   경고: nvidia-smi 미검출. GPU/CUDA 환경에서 실행하는지 확인하세요(CPU 추론은 느림)." >&2
fi

# --- 1. 가상환경 + 추론 의존성 ---------------------------------------------
echo "[1/2] 가상환경 준비: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null
# 파이프라인 의존성은 requirements-docker.txt 단일 소스에서 설치한다(그래프+벡터 전체 스택).
# Docker 이미지와 동일한 목록을 써서 venv/컨테이너 환경이 어긋나지 않도록 한다.
python -m pip install -r requirements-docker.txt

# --- 2. HuggingFace 가중치 사전 다운로드 -----------------------------------
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
  echo "[2/2] SKIP_MODEL_DOWNLOAD=1 → 가중치 다운로드 생략"
else
  echo "[2/2] HuggingFace 가중치 다운로드: $GRAPH_LLM_MODEL (캐시: ${HF_HOME:-$HOME/.cache/huggingface})"
  python - "$GRAPH_LLM_MODEL" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
path = snapshot_download(repo_id=repo_id)
print(f"   다운로드 완료: {path}")
PYEOF
fi

cat <<EOF

로컬 LLM 환경 구성 완료
   - venv     : $VENV_DIR   (활성화: source $VENV_DIR/bin/activate)
   - model    : $GRAPH_LLM_MODEL (HuggingFace 캐시에서 자동 로드)
   - HF 캐시  : ${HF_HOME:-$HOME/.cache/huggingface}   (Docker 는 이 경로를 볼륨 마운트)

빠른 확인:
   source $VENV_DIR/bin/activate
   python -c "from graph.graph_rag import build_local_llm; llm = build_local_llm(); print(llm.invoke('안녕').content)"
EOF

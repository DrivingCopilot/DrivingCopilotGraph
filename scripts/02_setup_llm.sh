#!/usr/bin/env bash
# 02_setup_llm.sh
# -----------------------------------------------------------------------------
# Graph RAG 엔티티/관계 추출에 사용할 로컬 LLM(Qwen3-VL-4B) 환경을 Ollama 로 구성한다.
#
# 구성 원칙:
#   - Ollama 서버/모델 : 네이티브 바이너리이므로 사용자 공간(~/.local)에 설치(sudo 불필요)
#   - Python 의존성     : 반드시 프로젝트 가상환경(.venv) 안에서만 설치
#
# 대상: GPU 워크스테이션(RTX 4060 Ti 8GB 등). qwen3-vl:4b 는 약 3.3GB(Q4)로
#       8GB VRAM 에서 비전 인코더/KV 캐시 포함 여유 있게 동작한다.
#
# 수행 내용:
#   1) Python 가상환경(.venv) 생성 + `ollama` 파이썬 클라이언트 설치
#   2) Ollama 서버 바이너리를 ~/.local 에 설치(없을 때만)
#   3) Ollama 서버 기동(백그라운드) 후 모델 pull
#
# 환경 변수(선택):
#   LLM_MODEL     내려받을 Ollama 모델 태그 (기본 qwen3-vl:4b)
#   VENV_DIR      가상환경 경로              (기본 .venv)
#   OLLAMA_HOME   Ollama 설치 경로           (기본 $HOME/.local)
#   OLLAMA_VERSION Ollama 릴리스 버전         (기본 v0.31.1)
#   SKIP_MODEL_PULL=1  모델 pull 생략(서버/의존성만 구성)
#
# 사용법:  ./scripts/02_setup_llm.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLM_MODEL="${LLM_MODEL:-qwen3-vl:4b}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OLLAMA_HOME="${OLLAMA_HOME:-$HOME/.local}"
OLLAMA_VERSION="${OLLAMA_VERSION:-v0.31.1}"
OLLAMA_BIN="$OLLAMA_HOME/bin/ollama"

# --- 0. 사전 점검 ----------------------------------------------------------
echo "[0/3] 환경 점검"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "!! $PYTHON_BIN 을 찾을 수 없습니다. Python 3.10+ 를 설치하세요." >&2
  exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
  echo "   경고: nvidia-smi 미검출. GPU/CUDA 환경에서 실행하는지 확인하세요(CPU 추론은 느림)."
fi

# --- 1. 가상환경 + 파이썬 클라이언트 --------------------------------------
echo "[1/3] 가상환경 준비: $VENV_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null
# Graph RAG 파이프라인 의존성:
#   - ollama         : Ollama 서버 호출 파이썬 클라이언트
#   - neo4j-graphrag : OllamaLLM / SchemaBuilder / LLMEntityRelationExtractor / Neo4jWriter
python -m pip install "ollama>=0.4" "neo4j-graphrag>=1.18"

# --- 2. Ollama 서버(사용자 공간 설치) -------------------------------------
echo "[2/3] Ollama 서버 확인: $OLLAMA_BIN"
if [[ ! -x "$OLLAMA_BIN" ]]; then
  echo "   Ollama 미설치 → $OLLAMA_HOME 에 설치($OLLAMA_VERSION)"
  TARBALL="/tmp/ollama-linux-amd64.tar.zst"
  URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
  curl -L --fail --progress-bar "$URL" -o "$TARBALL"
  mkdir -p "$OLLAMA_HOME"
  if command -v zstd >/dev/null 2>&1; then
    tar --use-compress-program=unzstd -C "$OLLAMA_HOME" -xf "$TARBALL"
  else
    # tar 가 zstd 를 직접 지원하는 최신 버전인 경우
    tar -C "$OLLAMA_HOME" -xf "$TARBALL"
  fi
  rm -f "$TARBALL"
fi
"$OLLAMA_BIN" --version >/dev/null 2>&1 || true
echo "   설치 경로: $OLLAMA_BIN"
echo "   PATH 추가 권장: export PATH=\"$OLLAMA_HOME/bin:\$PATH\""

# --- 3. 서버 기동 + 모델 pull ---------------------------------------------
if [[ "${SKIP_MODEL_PULL:-0}" == "1" ]]; then
  echo "[3/3] SKIP_MODEL_PULL=1 → 모델 pull 생략"
else
  echo "[3/3] Ollama 서버 기동 및 모델 pull: $LLM_MODEL"
  # 이미 떠 있으면 재사용, 아니면 백그라운드로 기동
  if ! curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "   서버 기동(백그라운드): ollama serve"
    OLLAMA_LOG="$ROOT_DIR/logs/ollama.log"
    mkdir -p "$ROOT_DIR/logs"
    nohup "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
    # 준비 대기(최대 30초)
    for _ in $(seq 1 30); do
      curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  "$OLLAMA_BIN" pull "$LLM_MODEL"
fi

cat <<EOF

✅ LLM 환경 구성 완료
   - venv     : $VENV_DIR   (활성화: source $VENV_DIR/bin/activate)
   - ollama   : $OLLAMA_BIN
   - model    : $LLM_MODEL

빠른 확인:
   export PATH="$OLLAMA_HOME/bin:\$PATH"
   ollama list
   ollama run $LLM_MODEL "안녕, 너는 이미지를 볼 수 있니?"

파이썬에서 사용:
   from ollama import Client
   c = Client()  # http://127.0.0.1:11434
   print(c.chat("$LLM_MODEL", messages=[{"role":"user","content":"테스트"}]))
EOF

#!/usr/bin/env bash
# 01_start_neo4j.sh
# -----------------------------------------------------------------------------
# Graph RAG용 Neo4j 로컬 서버를 기동한다.
#   - .env 가 없으면 .env.example 로부터 자동 생성
#   - docker compose 로 neo4j 컨테이너 기동
#   - healthcheck 가 통과(bolt 쿼리 가능)할 때까지 대기
#   - 실제 Cypher 쿼리로 접속 검증
#
# 사용법:  ./scripts/01_start_neo4j.sh
# 중지:    docker compose down   (데이터는 ./neo4j 볼륨에 유지됨)
# -----------------------------------------------------------------------------
set -euo pipefail

# 프로젝트 루트로 이동 (스크립트 위치 기준)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONTAINER="driving-copilot-neo4j"

# --- 1. .env 준비 ----------------------------------------------------------
if [[ ! -f .env ]]; then
  echo "[1/4] .env 가 없어 .env.example 로부터 생성합니다."
  cp .env.example .env
else
  echo "[1/4] 기존 .env 를 사용합니다."
fi

# .env 로드 (NEO4J_PASSWORD 등). 주석/빈 줄은 무시.
set -a
# shellcheck disable=SC1091
source .env
set +a
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

# --- 2. 컨테이너 기동 ------------------------------------------------------
echo "[2/4] Neo4j 컨테이너를 기동합니다 (docker compose up -d neo4j)."
docker compose up -d neo4j

# --- 3. healthcheck 대기 ---------------------------------------------------
echo "[3/4] Neo4j healthcheck 통과를 기다립니다 (최대 ~120초)..."
for i in $(seq 1 24); do
  status="$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo "unknown")"
  echo "    [$i] health: $status"
  if [[ "$status" == "healthy" ]]; then
    break
  fi
  if [[ "$status" == "unhealthy" ]]; then
    echo "!! 컨테이너가 unhealthy 상태입니다. 로그를 확인하세요: docker compose logs neo4j"
    exit 1
  fi
  sleep 5
done

if [[ "${status:-}" != "healthy" ]]; then
  echo "!! 시간 내에 healthy 상태가 되지 않았습니다. 로그: docker compose logs neo4j"
  exit 1
fi

# --- 4. 접속 검증 ----------------------------------------------------------
echo "[4/4] Cypher 쿼리로 접속을 검증합니다."
docker exec "$CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "RETURN 'Neo4j 접속 OK' AS status;"

cat <<EOF

✅ Neo4j 준비 완료
   - Browser : http://localhost:7474   (neo4j / $NEO4J_PASSWORD)
   - Bolt    : bolt://localhost:7687
   - 데이터   : ./neo4j/data (영속화)

다음 단계:
   - 테스트 데이터 적재 : ./scripts/test/run_cypher_test.sh
   - LLM 환경 설치      : ./scripts/02_setup_llm.sh
EOF

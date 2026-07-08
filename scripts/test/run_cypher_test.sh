#!/usr/bin/env bash
# run_cypher_test.sh
# -----------------------------------------------------------------------------
# 테스트용 Cypher 시드 데이터를 Neo4j 에 적재하고 결과를 검증한다.
# LLM 파이프라인 없이 그래프 스키마/탐색이 정상 동작하는지 확인하는 용도.
#
# 전제: ./scripts/01_start_neo4j.sh 로 Neo4j 가 기동되어 있어야 한다.
#
# 사용법:
#   ./scripts/test/run_cypher_test.sh          # 시드 적재 + 검증
#   ./scripts/test/run_cypher_test.sh --clean  # 테스트 데이터(_test=true) 삭제
# -----------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONTAINER="driving-copilot-neo4j"
SEED_FILE="scripts/test/seed_test_data.cypher"

# .env 로드 (접속 정보)
if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  source .env; set +a
fi
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

# cypher-shell 래퍼 (컨테이너 내부 실행)
run_cypher() {
  docker exec -i "$CONTAINER" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" "$@"
}

# Neo4j 접속 확인
if ! run_cypher "RETURN 1;" >/dev/null 2>&1; then
  echo "!! Neo4j 에 접속할 수 없습니다. 먼저 ./scripts/01_start_neo4j.sh 를 실행하세요." >&2
  exit 1
fi

# --- --clean: 테스트 데이터 삭제 -------------------------------------------
if [[ "${1:-}" == "--clean" ]]; then
  echo "▶ 테스트 데이터(_test=true) 삭제"
  run_cypher "MATCH (n {_test: true}) DETACH DELETE n;"
  echo "✅ 삭제 완료"
  exit 0
fi

# --- 시드 적재 -------------------------------------------------------------
echo "▶ 시드 적재: $SEED_FILE"
run_cypher < "$SEED_FILE"

# --- 검증 ------------------------------------------------------------------
echo "▶ 적재 결과 검증"
echo "  [노드 수 by 라벨]"
run_cypher "MATCH (n {_test: true}) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label;"

echo "  [관계 수 by 타입]"
run_cypher "MATCH (a {_test:true})-[r]->(b {_test:true}) RETURN type(r) AS rel, count(*) AS n ORDER BY rel;"

echo "  [탐색 샘플: DTC P0300 에서 1~2 hop]"
run_cypher "MATCH p=(d:\`DTC Code\` {code:'P0300'})-[*1..2]-(m)
RETURN d.code AS dtc,
       [rel IN relationships(p) | type(rel)] AS rels,
       coalesce(m.name, m.description, m.type, m.code) AS target
ORDER BY target;"

echo "✅ 테스트 완료. 정리하려면: ./scripts/test/run_cypher_test.sh --clean"

// seed_test_data.cypher
// -----------------------------------------------------------------------------
// Graph RAG 테스트용 시드 데이터.
// graph/schema.py(DrivingGraphSchema)의 노드/관계/패턴을 그대로 따른다.
// LLM 적재 파이프라인 없이 Neo4j 동작·탐색 쿼리를 검증하기 위한 용도.
//
// 시나리오: DTC P0300(랜덤 실화) → 엔진 경고등 점등 → 공회전 불안정 →
//           점화플러그 결함 → 점화플러그 교체(정비/조치)
//
// 모든 노드에 `_test: true` 마커를 부여해 정리(cleanup)를 쉽게 한다.
// 실행은 idempotent(MERGE 기반)하므로 여러 번 돌려도 중복되지 않는다.
// 라벨 `DTC Code`는 공백이 있어 백틱(``)으로 감싼다.
// -----------------------------------------------------------------------------

// --- 노드 -------------------------------------------------------------------
MERGE (sp:Component {name: 'spark plug'})
  SET sp.category = 'engine', sp._test = true;

MERGE (engine:System {name: 'engine'})
  SET engine._test = true;

MERGE (cel:WarningLight {name: 'check engine light'})
  SET cel.color = 'amber', cel._test = true;

MERGE (sym:Symptom {description: 'rough idle'})
  SET sym.severity = 6, sym._test = true;

MERGE (act:Action {description: 'replace spark plugs'})
  SET act.urgency = 'medium', act._test = true;

MERGE (mnt:Maintenance {type: 'spark plug replacement'})
  SET mnt.interval_months = 24, mnt.interval_miles = 60000, mnt._test = true;

MERGE (sch:Schedule {value: 60000, unit: 'miles'})
  SET sch._test = true;

MERGE (dtc:`DTC Code` {code: 'P0300'})
  SET dtc.description = 'Random/Multiple Cylinder Misfire Detected', dtc._test = true;

// --- 관계 (schema.py 의 patterns 준수) --------------------------------------
MATCH (sp:Component {name: 'spark plug'}),
      (engine:System {name: 'engine'}),
      (cel:WarningLight {name: 'check engine light'}),
      (sym:Symptom {description: 'rough idle'}),
      (act:Action {description: 'replace spark plugs'}),
      (mnt:Maintenance {type: 'spark plug replacement'}),
      (sch:Schedule {value: 60000, unit: 'miles'}),
      (dtc:`DTC Code` {code: 'P0300'})
MERGE (sp)-[:HAS_PART]->(engine)
MERGE (sp)-[:MAINTAINED_BY]->(mnt)
MERGE (cel)-[:INDICATES]->(sym)
MERGE (cel)-[:CAUSED_BY]->(sp)
MERGE (sym)-[:SYMPTOM_OF]->(sp)
MERGE (sym)-[:RESOLVED_BY]->(act)
MERGE (mnt)-[:APPLIES_TO]->(sp)
MERGE (mnt)-[:HAS_INTERVAL]->(sch)
MERGE (dtc)-[:MAPS_TO]->(sp)
MERGE (dtc)-[:TRIGGERS]->(cel);

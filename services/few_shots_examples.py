few_shot_examples = [
    {
        "category": "단순 조건 검색 (Filtering)",
        "question": "현재 미해결된 심각도 'High' 등급의 경고등이 있어?",
        "sql": "SELECT code, description FROM dtc_codes WHERE severity = 'High' AND resolved = 0;",
        "intent": "SQLite에서 BOOLEAN 값(resolved)을 0(False)과 1(True)로 처리하는 방법 학습"
    },
    {
        "category": "최신 데이터 조회 (Sorting & Limit)",
        "question": "가장 최근에 받은 정비 기록과 주행거리를 알려줘.",
        "sql": "SELECT date, type, mileage FROM maintenance_log ORDER BY date DESC LIMIT 1;",
        "intent": "ORDER BY DESC와 LIMIT 1을 조합하여 가장 최근(Max Date) 데이터를 가져오는 패턴 학습"
    },
    {
        "category": "패턴 매칭 및 집계 (LIKE & SUM)",
        "question": "지금까지 타이어 교체에 쓴 총비용은 얼마야?",
        "sql": "SELECT SUM(cost) FROM maintenance_log WHERE parts LIKE '%타이어%';",
        "intent": "특정 단어가 포함된 데이터를 찾을 때 LIKE 연산자를 쓰고, SUM 함수로 합계를 내는 방식 학습"
    },
    {
        "category": "시계열 집계 (Date Functions)",
        "question": "오늘 주행한 총 거리가 어떻게 돼?",
        "sql": "SELECT SUM(distance) FROM trip_history WHERE start_time >= date('now', 'localtime');",
        "intent": "SQLite 내장 함수인 date('now')를 활용해 '오늘'이라는 자연어 맥락을 시간 조건으로 변환하는 방법 학습"
    },
    {
        "category": "논리적 조인 / 서브쿼리 (Subquery)",
        "question": "가장 최근 운행했을 때의 최고 속도는 얼마였어?",
        "sql": "SELECT MAX(speed) FROM vehicle_telemetry WHERE ts >= (SELECT start_time FROM trip_history ORDER BY start_time DESC LIMIT 1);",
        "intent": "명시적인 외래 키(FK)가 없어도, 운행 기록(trip_history)의 시간 데이터를 활용해 센서 데이터(vehicle_telemetry)를 조회하는 의미론적 조인(Semantic Join) 학습"
    },
    {
        "category": "임계치 탐지 (Threshold Detection)",
        "question": "연료 잔량이 20% 이하로 떨어졌던 가장 최근 시간은 언제야?",
        "sql": "SELECT ts, fuel FROM vehicle_telemetry WHERE fuel <= 20.0 ORDER BY ts DESC LIMIT 1;",
        "intent": "부등호(<=) 연산과 실수형(REAL) 데이터 필터링 기준 학습"
    }
]
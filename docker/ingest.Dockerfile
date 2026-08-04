# docker/ingest.Dockerfile
#
# graph.ingest 적재 CLI 전용 이미지 (graph/ingest.py — PDF/txt → 엔티티·관계 추출 → Neo4j 적재).
# 상시 서비스가 아니라 일회성 배치 작업이다 — 이 레포의 유일한 상시 컨테이너는
# docker-compose.yml의 neo4j 뿐이고, 이 이미지는 그 옆에서 `docker compose run`으로
# 필요할 때만 띄운다.
#
# LLM 추출은 이 컨테이너 안이 아니라 호스트에 네이티브로 띄운 Ollama(scripts/02_setup_llm.sh,
# qwen3-vl:4b-instruct)가 담당한다 — GPU 컨테이너를 별도로 만들지 않고 OLLAMA_HOST로
# 호출만 한다(DrivingCopilotAgent 레포가 로컬 모델 서버를 별도 GPU 컨테이너로 분리한 것과
# 같은 이유: 이 이미지 자체는 GPU 텐서 연산을 하지 않는다).
#
# 빌드 (repo 루트에서):
#   docker build -f docker/ingest.Dockerfile -t driving-copilot-graph-ingest:latest .
# 실행: docker-compose.yml의 ingest 서비스로 (docker compose run --rm ingest ...) — README 참고.

FROM python:3.10-slim

WORKDIR /app

# sentence-transformers(PDF 입력 시 시맨틱 청킹용)가 의존성으로 torch를 끌고 오는데,
# 이 이미지는 CPU 전용이므로 CPU wheel을 먼저 명시적으로 깐다.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# graph/graph_rag.py, graph/schema.py, graph/ingest.py가 쓰는 SchemaBuilder /
# LLMEntityRelationExtractor / Neo4jWriter / OllamaLLM은 requirements.txt가 아니라
# scripts/02_setup_llm.sh가 별도 설치하는 패키지다(네이티브 개발 환경에서도 두 단계로
# 나뉘어 있는 설계) — 이미지에서도 동일하게 별도 스텝으로 설치한다.
RUN pip install --no-cache-dir "ollama>=0.4" "neo4j-graphrag>=1.18"

COPY core ./core
COPY graph ./graph
COPY services ./services

RUN useradd --create-home --uid 1000 appuser
USER appuser

ENTRYPOINT ["python", "-m", "graph.ingest"]

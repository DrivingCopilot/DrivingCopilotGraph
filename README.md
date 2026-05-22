# On-Device Multimodal Driving Copilot

**On-Device Multimodal Driving Copilot** 레포지토리에 오신 것을 환영합니다! 이 프로젝트는 LangChain과 LangGraph를 활용하여 복잡한 워크플로우, 추론 및 도구(Tool) 실행을 관리하는 고급 멀티모달 차량용 AI 비서를 구현합니다.

## 레포지토리 구조

- **`graph/`**: 핵심 상태 그래프(StateGraph) 정의 및 RAG 구현체를 포함합니다.
  - `graph_rag.py`
  - `vector_rag.py`
  - *(예정)* Text2SQL 에이전트 구현체.
- **`services/`**: 시맨틱 청커(Semantic Chunker), 임베더(Embedder), PDF 파서(Parser)와 같은 서비스 모듈 및 데이터 처리 스크립트를 보관합니다.

## 추가 예정 사항

**Knowledge Agent MCP Tools**: Knowledge Agent를 위한 포괄적인 MCP(Model Context Protocol) 도구들을 관리하고 보관할 전용 폴더가 향후 생성될 예정입니다.
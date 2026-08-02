import asyncio
import logging
from typing import TypedDict, Any, List, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j import AsyncGraphDatabase, GraphDatabase
from neo4j_graphrag.experimental.components.schema import SchemaBuilder
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import (
    TextChunk,
    TextChunks,
    DocumentInfo,
)
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import LLMBase, LLMResponse
from pydantic import BaseModel, Field

from core import config
from graph.schema import DrivingGraphSchema

logger = logging.getLogger(__name__)


# --- 1. Local LLM (HuggingFace / Qwen3-VL) ----------------------------------
class LocalQwen3VL(LLMBase):
    """
    HuggingFace 에서 내려받은 로컬 Qwen3-VL 가중치를 transformers 로 직접 로드해 추론한다.

    neo4j_graphrag.llm.LLMBase 를 구현하므로
    - LLMEntityRelationExtractor(llm=...) 에 그대로 주입 가능하고
    - invoke/ainvoke → LLMResponse(.content) 형태로 응답한다.

    Graph RAG 추출/융합은 이미지가 아닌 텍스트만 다루므로 비전 입력 없이
    텍스트 전용 채팅 템플릿으로 호출한다.

    사전 준비: scripts/02_setup_llm.sh 로 모델 가중치 다운로드 완료 필요.
    """

    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict] = None,
        device: str = "auto",
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, model_params=model_params)

        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        # device="auto" 는 GPU 가 있으면 cuda, 없으면 cpu 로 해석한다(비-GPU 환경/CI 에서도
        # 임포트·로딩이 죽지 않도록). "cuda"/"cpu" 등 명시값은 그대로 device_map 에 전달한다.
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.temperature = (self.model_params.get("options") or {}).get(
            "temperature", 0.0
        )
        self.max_new_tokens = max_new_tokens
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else None,
        )
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output[0] if output else ""

    def invoke(
        self,
        input: str,
        message_history=None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if not isinstance(input, str):
            raise NotImplementedError("LocalQwen3VL only supports text(str) input")
        try:
            return LLMResponse(content=self._generate(input, system_instruction))
        except Exception as e:
            raise LLMGenerationError(e)

    async def ainvoke(
        self,
        input: str,
        message_history=None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # transformers.generate 는 동기/블로킹이므로 스레드로 오프로딩해 이벤트 루프를 막지 않는다.
        return await asyncio.to_thread(self.invoke, input, message_history, system_instruction)


def build_local_llm() -> LocalQwen3VL:
    """
    HuggingFace 에서 내려받은 로컬 Qwen3-VL(Instruct) 가중치로 LLM 인스턴스를 생성한다.

    사전 준비: scripts/02_setup_llm.sh 로 모델 가중치 다운로드 완료 필요.
    """
    return LocalQwen3VL(
        model_name=config.GRAPH_LLM_MODEL,
        model_params={"options": {"temperature": config.GRAPH_LLM_TEMPERATURE}},
        device=config.GRAPH_LLM_DEVICE,
        max_new_tokens=config.GRAPH_LLM_MAX_NEW_TOKENS,
    )


# --- 3. MCP Tool Interface (Input/Output Schemas) ---
class ExtractionInput(BaseModel):
    text: str = Field(..., description="Vehicle diagnostic text (DTC, symptoms) to extract entities from.")

class ExtractionOutput(BaseModel):
    success: bool
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    context: str
    error: Optional[str] = None

# --- Main Graph RAG Component ---
class VehicleGraphManager:
    """
    Manages Neo4j Knowledge Graph operations for the Vehicle Copilot.
    Can be used as a LangGraph Node ('Knowledge Agent') or registered as an MCP Tool.
    """
    def __init__(self):
        self.uri = config.NEO4J_URI
        self.user = config.NEO4J_USER
        self.password = config.NEO4J_PASSWORD
        self.database = config.NEO4J_DATABASE
        
        # Async Driver: 비동기 그래프 탐색(retrieve_context)용
        self.driver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password)
        )
        # Sync Driver: Neo4jWriter 는 내부적으로 동기 execute_query 를 호출하므로 별도 필요
        self.sync_driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )

        # Build Schema Pipeline
        # neo4j_graphrag>=1.18: SchemaBuilder 는 인자 없이 생성하고,
        # node/relationship/pattern 은 run() 시점에 전달한다.
        self.schema_builder = SchemaBuilder()

        # 로컬 HuggingFace LLM (Qwen3-VL) — 추출/융합에서 공유
        self.llm = build_local_llm()

        # LLM Extractor — 기본 ERExtractionTemplate 사용(JSON 출력 포맷 지시 포함).
        # 커스텀 프롬프트는 JSON 포맷 명세가 없어 구조화 추출이 실패하므로 사용하지 않는다.
        self.extractor = LLMEntityRelationExtractor(
            llm=self.llm,
            create_lexical_graph=True,
        )

        # Neo4j 적재기 — clean_db=False 로 증분 적재(매 호출 시 DB 초기화 방지)
        self.writer = Neo4jWriter(
            self.sync_driver,
            neo4j_database=self.database,
            clean_db=False,
        )

        # 병렬 적재 지원:
        #  - _graph_schema : 스키마는 불변이므로 최초 1회만 빌드해 재사용(청크마다 재빌드 방지)
        #  - _write_lock   : 여러 청크가 같은 Document/Chunk 노드를 동시에 MERGE 하면
        #                    Neo4j 데드락이 나므로, 추출은 병렬로 두되 쓰기만 직렬화한다.
        self._graph_schema = None
        self._write_lock = asyncio.Lock()

    async def close(self):
        """Close both Neo4j driver connections."""
        await self.driver.close()
        self.sync_driver.close()

    @retry(
        stop=stop_after_attempt(3), # Initial try + 2 retries = 3 attempts total
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def extracting_data(
        self,
        text: str,
        document_info: Optional[DocumentInfo] = None,
        store: bool = True,
    ) -> dict:
        """
        차량 진단 텍스트에서 엔티티/관계를 추출하고(store=True 시) Neo4j 에 적재한다.

        Args:
            text: 추출 대상 텍스트(예: 매뉴얼 청크 page_content).
            document_info: 출처 메타데이터(source/파일명 등). 지정 시 Document 노드로 기록.
            store: True 면 추출 결과를 Neo4j 에 적재. False 면 추출만 수행.

        Tenacity 로 최대 2회 재시도한다.
        """
        try:
            # 1. GraphSchema 는 불변이므로 최초 1회만 빌드해 캐시(병렬 호출 시 이중 빌드는 무해).
            if self._graph_schema is None:
                self._graph_schema = await self.schema_builder.run(
                    node_types=DrivingGraphSchema.get_node_types(),
                    relationship_types=DrivingGraphSchema.get_relationship_types(),
                    patterns=DrivingGraphSchema.get_patterns(),
                )
            graph_schema = self._graph_schema

            # 2. 입력 텍스트를 TextChunks 로 감싸 추출기에 전달 (LLM 추출 — 병렬 가능한 느린 구간)
            chunks = TextChunks(chunks=[TextChunk(index=0, text=text)])
            graph = await self.extractor.run(
                chunks=chunks,
                schema=graph_schema,
                document_info=document_info,
            )

            # 3. Neo4j 적재 (Neo4jWriter.run 은 async 컴포넌트)
            #    같은 Document/Chunk 노드 동시 MERGE 로 인한 데드락 방지를 위해 쓰기는 직렬화.
            if store:
                async with self._write_lock:
                    await self.writer.run(graph)

            return {
                "success": True,
                "data": graph.model_dump() if hasattr(graph, "model_dump") else graph,
            }
        except Exception as e:
            logger.error(f"Failed to extract and store graph data: {e}")
            raise # Triggers Tenacity retry mechanism

    async def retrieve_context(self, entities: List[Dict[str, Any]]) -> str:
        """
        추출된 Entity들을 기반으로 Neo4j에서 1~2 hop 그래프 탐색을 수행하여
        관련 지식(Context)을 확보합니다.
        """
        if not entities:
            return "No entities provided for graph traversal."

        # 추출된 엔티티들의 식별자 목록 추출 (대소문자 무관 탐색을 위해 소문자화).
        # 노드 종류마다 식별 속성이 다르다(Component=name, DTC Code=code, Symptom=description,
        # System=name 등). name 만 보면 Symptom/DTC 를 놓치므로 code/type/description 도 함께 본다.
        entity_names = []
        for entity in entities:
            props = entity.get("properties", {}) or {}
            name = (
                props.get("name")
                or props.get("code")
                or props.get("type")
                or props.get("description")
                or entity.get("id", "")
            )
            if name:
                entity_names.append(str(name).lower())

        if not entity_names:
            return "Could not identify valid entity names for traversal."

        # 1~2 hop 탐색 Cypher 쿼리 (가변 경로 탐색).
        # 노드별 식별 속성 차이를 고려해 name/id/code/type/description 을 모두 매칭 대상으로 삼는다
        # (toLower(null)=null 이라 존재하지 않는 속성은 자연히 매칭에서 제외된다).
        # 출력 이름도 coalesce 로 노드 종류에 맞는 식별값을 표시한다.
        cypher_query = """
        MATCH p = (n)-[*1..2]-(m)
        WHERE toLower(n.name) IN $entity_names
           OR toLower(n.id) IN $entity_names
           OR toLower(n.code) IN $entity_names
           OR toLower(n.type) IN $entity_names
           OR toLower(n.description) IN $entity_names
        RETURN coalesce(n.name, n.code, n.type, n.description, n.id) AS source_name,
               labels(n) AS source_labels,
               [rel IN relationships(p) | type(rel)] AS rel_types,
               coalesce(m.name, m.code, m.type, m.description, m.id) AS target_name,
               labels(m) AS target_labels
        LIMIT $limit
        """

        try:
            # 비동기 세션을 열고 쿼리 실행
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    cypher_query,
                    entity_names=entity_names,
                    limit=config.GRAPH_MAX_RESULTS,
                )
                records = await result.data()

            if not records:
                return f"No related context found in graph for entities: {', '.join(entity_names)}"

            # 검색된 경로들을 자연어 문장으로 변환하여 Context Fusion 준비
            sentences = []
            for record in records:
                source = record.get('source_name', 'Unknown')
                target = record.get('target_name', 'Unknown')
                rel_types = record.get('rel_types', [])
                
                # 경로상의 관계들을 문자열로 연결 (예: HAS_PART -> CAUSES)
                rel_chain = " -> ".join([str(r).replace("_", " ") for r in rel_types])
                
                sentences.append(f"Graph Path: {source} [{rel_chain}] {target}.")
                
            return " ".join(sentences)

        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return f"Error during graph traversal: {e}"

    async def fuse_contexts(self, query: str, vector_context: str, graph_context: str) -> str:
        """
        Vector RAG(차량 매뉴얼)와 Graph RAG(구조화된 진단 지식)을 모두 사용하여 융합할 때 사용(추후 Supervisor agent에서 따로 호출 가능)
        """


        # 지식 충돌 방지를 위한 프롬프트 엔지니어링
        fusion_prompt = f"""
You are an expert vehicle diagnostic assistant.
Your task is to synthesize a unified and highly accurate diagnostic context by fusing unstructured manual data (Vector RAG) and structured relationships (Graph RAG).
If there is a conflict, prioritize the structural facts from the Graph DB, but enrich it with the step-by-step procedures from the Vector DB.

[User Query / Symptom]
{query}

[Vector RAG Context (Unstructured Manuals)]
{vector_context}

[Graph RAG Context (Structured Knowledge paths)]
{graph_context}

Please provide the synthesized diagnostic context.
"""
        try:
            # 비동기 호출을 통해 LLM 텍스트 생성 (LLMResponse.content 반환)
            response = await self.llm.ainvoke(fusion_prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Context fusion failed: {e}")
            return f"Failed to fuse context. Fallback -> Vector: {vector_context[:100]} | Graph: {graph_context[:100]}"

    async def mcp_run_extraction(self, input_data: ExtractionInput) -> ExtractionOutput:
        """
        MCP Tool wrapper: Exposes extraction logic as an MCP-compatible interface.
        """
        try:
            # 1. 텍스트에서 Entity와 Relationship 추출
            result = await self.extracting_data(input_data.text)
            ext_data = result.get("data", {})
            
            # extracting_data 는 graph.model_dump() 을 반환하며 키는 nodes/relationships 다
            # (graph/ingest.py 도 data.get("nodes") 를 사용). "entities" 로 읽으면 항상 빈 값이 된다.
            entities = ext_data.get("nodes", []) if isinstance(ext_data, dict) else []
            relationships = ext_data.get("relationships", []) if isinstance(ext_data, dict) else []
            
            # 2. 방금 추출된 Entity를 단서로 Neo4j 그래프 탐색 (1~2 hop)
            traversal_context = await self.retrieve_context(entities)
            

            # JSON 형식으로 데이터 반환
            return ExtractionOutput(
                success=True,
                entities=entities,
                relationships=relationships,
                context=traversal_context
            )
        except Exception as e:
            return ExtractionOutput(
                success=False,
                entities=[],
                relationships=[],
                context="",
                error=str(e)
            )
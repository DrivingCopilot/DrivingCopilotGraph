import os
import asyncio
import logging
from typing import TypedDict, Any, List, Dict, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from neo4j import AsyncGraphDatabase
from neo4j_graphrag.experimental.components.schema import SchemaBuilder
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from app.agent.state import AgentState

from app.graph.schema import DrivingGraphSchema

logger = logging.getLogger(__name__)

# --- 1. Abstracted Local LLM Interface ---
class LocalQwen2VL(BaseChatModel):
    model_name: str = "qwen2-vl-1.5b-instruct-int4"
    
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("Requires local TensorRT-LLM binding implementation")
        
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # Implementation for invoking local TensorRT-LLM asynchronously goes here
        raise NotImplementedError("Requires local TensorRT-LLM binding implementation")
        
    @property
    def _llm_type(self) -> str:
        return "local-qwen2-vl-1.5b-executor"


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
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Async Driver setup for non-blocking I/O
        self.driver = AsyncGraphDatabase.driver(
            self.uri, 
            auth=(self.user, self.password)
        )
        
        # Build Schema Pipeline
        self.schema_builder = SchemaBuilder(
            node_types=DrivingGraphSchema.get_node_types(),
            relationship_types=DrivingGraphSchema.get_relationship_types(),
            patterns=DrivingGraphSchema.get_patterns()
        )

        # 프롬프트 템플릿에 {schema} 변수 추가
        extraction_prompt = """
You are a top-tier automotive data extraction expert.
Extract entities and relationships from the following vehicle diagnostic text strictly based on the provided schema.

[ALLOWED SCHEMA]
{schema}

[TEXT TO EXTRACT]
{text}
"""

        # LLM Extractor using the abstracted Local LLM
        self.extractor = LLMEntityRelationExtractor(
            llm=LocalQwen2VL(),
            prompt_template=extraction_prompt,
            create_lexical_graph=True
        )

    async def close(self):
        """Close the async Neo4j driver connection."""
        await self.driver.close()

    @retry(
        stop=stop_after_attempt(3), # Initial try + 2 retries = 3 attempts total
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def extracting_data(self, text: str) -> dict:
        """
        Async extraction of entities and relationships, and storage to Neo4j.
        Includes failure handling with tenacity (max 2 retries).
        """
        try:
            # 1. SchemaBuilder를 실행하여 텍스트 형태의 스키마를 가져옴
            schema_info = await asyncio.to_thread(self.schema_builder.run)
            schema_text = schema_info.schema if hasattr(schema_info, 'schema') else str(schema_info)

            # 2. 텍스트와 추출된 스키마 텍스트를 함께 전달하여 실행
            extraction_result = await asyncio.to_thread(
                self.extractor.run, 
                text=text,
                schema=schema_text
            )

            return {
                "success": True,
                "data": getattr(extraction_result, "dict", lambda: extraction_result)()
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

        # 추출된 엔티티들의 이름(ID) 목록 추출 (대소문자 무관 탐색을 위해 소문자화)
        entity_names = []
        for entity in entities:
            name = entity.get("properties", {}).get("name", entity.get("id", ""))
            if name:
                entity_names.append(str(name).lower())

        if not entity_names:
            return "Could not identify valid entity names for traversal."

        # 1~2 hop 탐색 Cypher 쿼리 (가변 경로 탐색)
        cypher_query = """
        MATCH p = (n)-[*1..2]-(m)
        WHERE toLower(n.name) IN $entity_names OR toLower(n.id) IN $entity_names
        RETURN n.name AS source_name, labels(n) AS source_labels,
               [rel IN relationships(p) | type(rel)] AS rel_types,
               m.name AS target_name, labels(m) AS target_labels
        LIMIT 20
        """
        
        try:
            # 비동기 세션을 열고 쿼리 실행
            async with self.driver.session() as session:
                result = await session.run(cypher_query, entity_names=entity_names)
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
            llm = LocalQwen2VL()
            # 비동기 호출을 통해 LLM 텍스트 생성
            response = await llm.ainvoke(fusion_prompt)
            # LangChain BaseMessage 형태 반환 시 content 추출
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
            
            entities = ext_data.get("entities", []) if isinstance(ext_data, dict) else []
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
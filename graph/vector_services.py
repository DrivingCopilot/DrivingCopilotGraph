"""
graph/vector_services.py

Vector RAG에 필요한 서비스(PDF 파서, 시맨틱 청커, 임베더)를 통합한 모듈.
- VehiclePDFParser: PDF를 파싱하여 Document 리스트 생성
- SemanticChunker: Document를 의미 단위로 청킹
- VehicleEmbedder: 청크를 임베딩하여 Qdrant에 저장
"""

from __future__ import annotations  # Python 3.9 이하에서도 타입 힌트가 동작하도록 함

# 표준 라이브러리
import logging
import re
from pathlib import Path

# 서드파티 라이브러리
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# 내부 애플리케이션 임포트
from app.core.config import (
    MODEL_NAME,
    VECTOR_SIZE,
    QDRANT_PATH,
    COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

# --- From services/pdf_parser.py ---

def _clean_text(raw: str) -> str:
    """
    PDF에서 추출한 원본 텍스트의 노이즈를 제거하고 공백을 정규화한다.
    줄 단위로 순회하며 불필요한 패턴을 필터링한다.
    """
    cleaned = []  # 정제된 줄을 담을 리스트

    for line in raw.splitlines():  # 텍스트를 줄 단위로 분리하여 순회

        # 목차의 점선 제거 (예: "차량 외부 .............. 7")
        if re.search(r'\.{5,}', line):
            continue

        # 단독 페이지 번호 제거 (예: "123")
        if re.match(r'^\s*\d+\s*$', line):
            continue

        # 구분선 제거 (예: "-----", "=====", "_____")
        if re.match(r'^\s*[-_=]{5,}\s*$', line):
            continue

        # 이미지 파일명 패턴 제거 (예: "OutsideVehicleFrontOverview")
        if re.match(r'^[A-Z0-9][A-Za-z0-9_]+$', line.strip()):
            continue

        # 번호만 있는 라인 제거 (예: "1:", "2:", "3:")
        if re.match(r'^\s*\d+:\s*$', line):
            continue
        
        line = re.sub(r'\ufffd+', '', line)
        line = re.sub(r'\(cid:\d+\)', '', line)
        line = re.sub(r'[□■▣]{2,}', '', line)
        line = re.sub(r'[Ƅ-ƿ]+', '', line)
        line = re.sub(r'[\u0a80-\u0aff]+', '', line)
        line = re.sub(r'[\u0980-\u09ff]+', '', line)
        line = re.sub(r'[\u0600-\u06ff]+', '', line)
        line = re.sub(r'[\u0400-\u04ff]+', '', line)
        line = re.sub(r'[\u0e00-\u0e7f]+', '', line)

        if line.strip():
            cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _estimate_body_font_size(fitz_doc: fitz.Document) -> float:
    """
    문서 전체를 순회하여 가장 많이 사용된 폰트 크기를 본문 크기로 추정한다.
    """
    size_counts: dict[float, int] = {}

    for page in fitz_doc:
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span.get("size", 0), 1)
                    size_counts[size] = size_counts.get(size, 0) + 1

    return max(size_counts, key=size_counts.__getitem__) if size_counts else 10.0


def _detect_section(fitz_page: fitz.Page, body_size: float) -> str:
    """
    페이지에서 본문보다 큰 폰트의 텍스트를 섹션 헤더로 감지한다.
    """
    for block in fitz_page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0)

                if size > body_size:
                    if re.search(r'[가-힣a-zA-Z]', text):
                        return text

    return ""


class VehiclePDFParser:
    """
    차량 매뉴얼 PDF 파서.
    """

    def __init__(self, min_text_length: int = 20) -> None:
        self.min_text_length = min_text_length

    def parse(self, pdf_path: str | Path) -> list[Document]:
        """
        PDF를 파싱하여 LangChain Document 리스트를 반환한다.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        documents = []
        lc_pages = PyMuPDFLoader(str(pdf_path)).load()

        with fitz.open(str(pdf_path)) as fitz_doc:
            body_size = _estimate_body_font_size(fitz_doc)
            current_section = ""

            for lc_page, fitz_page in zip(lc_pages, fitz_doc):
                page_num = lc_page.metadata.get("page", 0) + 1
                detected = _detect_section(fitz_page, body_size)
                if detected:
                    current_section = detected

                content = _clean_text(lc_page.page_content)

                if not content or len(content) < self.min_text_length:
                    continue

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": pdf_path.name,
                        "page_num": page_num,
                        "section": current_section,
                        "content_type": "text",
                    }
                ))

        return documents

# --- From services/semantic_chunker.py ---

class SemanticChunker:
    """
    LangChain SemanticChunker 기반 청커.
    """

    def __init__(
        self,
        breakpoint_threshold_type: str = "standard_deviation",
        embeddings: HuggingFaceEmbeddings | None = None,
    ) -> None:
        self._embeddings = embeddings or HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._chunker = LangChainSemanticChunker(
            embeddings=self._embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Document 리스트를 시맨틱 경계에서 분할한다.
        """
        chunks = self._chunker.split_documents(documents)
        chunks = [c for c in chunks if c.page_content.strip()]
        chunks = self._merge_small(chunks, min_tokens=64)
        return chunks

    def _merge_small(self, chunks: list[Document], min_tokens: int = 64) -> list[Document]:
        """min_tokens 미만 청크를 앞 청크에 병합."""
        if not chunks:
            return chunks
        merged = [chunks[0]]
        for cur in chunks[1:]:
            prev = merged[-1]
            prev_tokens = len(prev.page_content.split())
            if (prev.metadata.get("source") == cur.metadata.get("source")
                    and prev_tokens < min_tokens):
                prev.page_content += " " + cur.page_content
            else:
                merged.append(cur)
        return merged

# --- From services/embedder.py ---

class VehicleEmbedder:
    """
    LangChain Qdrant 기반 임베딩 및 벡터스토어 관리 클래스.
    """

    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings | None = None,
    ) -> None:
        self._embeddings = embeddings or HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._client = QdrantClient(path=QDRANT_PATH)
        self._ensure_collection()
        self._vectorstore = QdrantVectorStore(
            client=self._client,
            collection_name=COLLECTION_NAME,
            embedding=self._embeddings,
        )

    def embed_and_store(self, documents: list[Document], batch_size: int = 32) -> None:
        """
        Document 리스트를 임베딩하고 Qdrant에 저장한다.
        """
        if not documents:
            return

        for i in range(0, len(documents), batch_size):
            batch = documents[i: i + batch_size]
            self._vectorstore.add_documents(batch)
            logger.info("임베딩 진행: %d / %d", min(i + batch_size, len(documents)), len(documents))

        logger.info("완료: %d개 Document Qdrant 저장", len(documents))

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """
        Qdrant에서 쿼리와 유사한 Document를 검색한다.
        """
        return self._vectorstore.similarity_search(query, k=top_k)

    def search_with_score(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """
        점수와 함께 검색한다.
        """
        return self._vectorstore.similarity_search_with_score(query, k=top_k)

    def _ensure_collection(self) -> None:
        """
        Qdrant 컬렉션이 없으면 생성하고 payload 인덱스를 추가한다.
        """
        existing = [c.name for c in self._client.get_collections().collections]

        if COLLECTION_NAME in existing:
            return

        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE,
                distance=qmodels.Distance.COSINE,
            ),
            quantization_config=qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            ),
        )

        for field_name in ("source", "section", "content_type"):
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )

        logger.info("Qdrant 컬렉션 생성: %s", COLLECTION_NAME)
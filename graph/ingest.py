"""
graph/ingest.py

차량 매뉴얼(PDF) 또는 진단 텍스트(.txt)를 청킹 → 엔티티/관계 추출 → Neo4j 적재하는
엔트리포인트. scripts/03_ingest.sh 가 `python -m graph.ingest <입력>` 형태로 호출한다.

설계 원칙 — 단일 원본 공유:
    Vector RAG(Qdrant) 인덱싱과 **동일한** VehiclePDFParser + SemanticChunker 를 재사용한다.
    즉 Qdrant 에 들어가는 청크와 같은 청크가 Neo4j 그래프로도 적재되어, 벡터/그래프가
    같은 원본 텍스트를 공유한다.

    PDF ─▶ VehiclePDFParser.parse ─▶ SemanticChunker.chunk ─▶ list[Document]
        └─(각 청크 page_content)─▶ VehicleGraphManager.extracting_data ─▶ Neo4j

.txt 입력:
    임베딩 스택(torch/sentence-transformers) 없이도 그래프 적재를 시험할 수 있도록,
    빈 줄 기준으로 문단 단위 분할하여 그대로 추출에 사용한다.

Usage:
    python -m graph.ingest <파일 또는 디렉터리>
    python -m graph.ingest data/manuals/            # 디렉터리 내 모든 pdf/txt
    python -m graph.ingest manual.pdf --max-chunks 20   # 앞 20개 청크만(스모크 테스트)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from neo4j_graphrag.experimental.components.types import DocumentInfo

from core import config
from graph.graph_rag import VehicleGraphManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
# 라이브러리 INFO 소음 억제(Neo4j 알림/httpx 요청 로그)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("graph.ingest")

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


# --- 입력 파일 수집 --------------------------------------------------------
def iter_input_files(path: Path) -> List[Path]:
    """파일이면 그 파일을, 디렉터리면 하위의 pdf/txt 를 정렬해 반환."""
    if path.is_file():
        return [path]
    files = sorted(
        p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return files


# --- 청킹 ------------------------------------------------------------------
def chunks_from_pdf(pdf_path: Path) -> List[Tuple[str, dict]]:
    """
    Qdrant 파이프라인과 동일하게 PDF 를 파싱/시맨틱 청킹한다.
    반환: [(청크 텍스트, 메타데이터dict), ...]

    NOTE: SemanticChunker 는 bge-m3(sentence-transformers/torch) 를 로드하므로
          해당 의존성이 설치된 환경에서만 동작한다. 지연 임포트로 처리.
    """
    from services.pdf_parser import VehiclePDFParser
    from services.semantic_chunker import SemanticChunker

    documents = VehiclePDFParser().parse(str(pdf_path))
    chunks = SemanticChunker().chunk(documents)
    return [(c.page_content, dict(c.metadata)) for c in chunks]


def chunks_from_txt(txt_path: Path) -> List[Tuple[str, dict]]:
    """
    .txt 는 빈 줄(문단) 기준으로 분할한다(임베딩 스택 불필요).
    반환: [(문단 텍스트, 메타데이터dict), ...]
    """
    raw = txt_path.read_text(encoding="utf-8")
    paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return [
        (p, {"source": txt_path.name, "chunk_index": i})
        for i, p in enumerate(paras)
    ]


def chunks_from_qdrant(
    collection: str,
    qdrant_path: str,
) -> List[Tuple[str, dict]]:
    """
    Qdrant 로컬 컬렉션(벡터 RAG 인덱싱 결과)의 청크를 그대로 그래프 원본으로 재사용한다.
    벡터/그래프가 **같은 청크**를 공유하도록 하는 단일 원본 경로(원본 PDF·torch 불필요).

    payload 규약: {"page_content": <본문>, "metadata": {source, page_num, section, ...}}
    반환: [(청크 텍스트, 메타데이터dict), ...] — 파일 경로 기반 경로와 동일한 형태.
    """
    from qdrant_client import QdrantClient

    client = QdrantClient(path=qdrant_path)
    try:
        out: List[Tuple[str, dict]] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                text = payload.get("page_content", "")
                if not text or not text.strip():
                    continue
                meta = dict(payload.get("metadata", {}) or {})
                meta.setdefault("qdrant_id", str(p.id))
                out.append((text, meta))
            if offset is None:
                break
        return out
    finally:
        client.close()


def load_chunks(file_path: Path) -> List[Tuple[str, dict]]:
    if file_path.suffix.lower() == ".pdf":
        return chunks_from_pdf(file_path)
    return chunks_from_txt(file_path)


# --- 적재 ------------------------------------------------------------------
async def _ingest_chunks(
    manager: VehicleGraphManager,
    chunks: List[Tuple[str, dict]],
    default_path: str,
    concurrency: int = 1,
) -> Tuple[int, int, int, int, int]:
    """
    청크 목록을 추출→적재하고 (총계, 성공, 실패, 노드누적, 관계누적) 을 반환한다.

    concurrency>1 이면 세마포어로 동시에 N개 청크를 처리한다. 느린 LLM 추출은 병렬로
    겹쳐 실행되고, Neo4j 쓰기는 manager 내부 _write_lock 으로 직렬화되어 데드락을 막는다.
    DocumentInfo.path 는 청크 메타의 source(원본 PDF명)를 우선 사용해 원본별 Document 노드를 만든다.
    """
    n_all = len(chunks)
    sem = asyncio.Semaphore(max(1, concurrency))
    tally = {"total": 0, "ok": 0, "fail": 0, "nodes": 0, "rels": 0}

    async def worker(idx: int, text: str, meta: dict) -> None:
        # DocumentInfo.metadata 는 Dict[str, str] 이어야 하므로 값을 문자열화
        doc_info = DocumentInfo(
            path=str(meta.get("source", default_path)),
            metadata={k: str(v) for k, v in meta.items()},
        )
        async with sem:
            tally["total"] += 1
            try:
                res = await manager.extracting_data(text, document_info=doc_info)
                data = res.get("data", {})
                n = len(data.get("nodes", []))
                r = len(data.get("relationships", []))
                tally["nodes"] += n
                tally["rels"] += r
                tally["ok"] += 1
                logger.info("  [%d/%d] nodes=%d rels=%d", idx + 1, n_all, n, r)
            except Exception as e:
                tally["fail"] += 1
                logger.warning("  [%d/%d] 추출/적재 실패: %s", idx + 1, n_all, e)

    await asyncio.gather(
        *(worker(i, text, meta) for i, (text, meta) in enumerate(chunks))
    )
    return tally["total"], tally["ok"], tally["fail"], tally["nodes"], tally["rels"]


async def ingest_files(
    files: Iterable[Path],
    max_chunks: int | None = None,
    concurrency: int = 1,
) -> None:
    manager = VehicleGraphManager()
    total_chunks = ok_chunks = fail_chunks = 0
    total_nodes = total_rels = 0
    try:
        for file_path in files:
            logger.info("▶ 파일 처리: %s", file_path)
            try:
                chunks = load_chunks(file_path)
            except ModuleNotFoundError as e:
                logger.error(
                    "청킹 의존성 누락(%s). PDF 처리는 sentence-transformers/torch 설치 필요. "
                    "우선 .txt 로 시험하거나 GPU 환경에서 실행하세요.",
                    e.name,
                )
                continue

            if max_chunks is not None:
                chunks = chunks[:max_chunks]
            logger.info("  청크 수: %d (동시성 %d)", len(chunks), concurrency)

            t, o, f, n, r = await _ingest_chunks(
                manager, chunks, str(file_path), concurrency=concurrency
            )
            total_chunks += t; ok_chunks += o; fail_chunks += f
            total_nodes += n; total_rels += r
    finally:
        await manager.close()

    logger.info(
        "✅ 적재 완료 — 청크 %d개(성공 %d / 실패 %d) | 누적 노드 %d, 관계 %d",
        total_chunks, ok_chunks, fail_chunks, total_nodes, total_rels,
    )


async def ingest_qdrant(
    collection: str,
    qdrant_path: str,
    max_chunks: int | None = None,
    concurrency: int = 1,
) -> None:
    """Qdrant 로컬 컬렉션의 청크를 그대로 그래프로 적재(벡터/그래프 단일 원본 공유)."""
    logger.info("▶ Qdrant 컬렉션 처리: %s (path=%s)", collection, qdrant_path)
    chunks = chunks_from_qdrant(collection, qdrant_path)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    logger.info("  청크 수: %d (동시성 %d)", len(chunks), concurrency)

    manager = VehicleGraphManager()
    try:
        t, o, f, n, r = await _ingest_chunks(
            manager, chunks, collection, concurrency=concurrency
        )
    finally:
        await manager.close()

    logger.info(
        "✅ 적재 완료 — 청크 %d개(성공 %d / 실패 %d) | 누적 노드 %d, 관계 %d",
        t, o, f, n, r,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF/텍스트/Qdrant → Neo4j 그래프 적재")
    parser.add_argument(
        "input",
        nargs="?",
        help="입력 파일 또는 디렉터리 (pdf/txt). --qdrant 사용 시 생략 가능",
    )
    parser.add_argument(
        "--qdrant",
        action="store_true",
        help="파일 대신 Qdrant 로컬 컬렉션의 청크를 그래프로 적재",
    )
    parser.add_argument(
        "--collection",
        default=config.COLLECTION_NAME,
        help=f"Qdrant 컬렉션명 (기본 {config.COLLECTION_NAME})",
    )
    parser.add_argument(
        "--qdrant-path",
        default="services/qdrant_storage",
        help="Qdrant 로컬 저장소 경로 (기본 services/qdrant_storage)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="처리할 최대 청크 수(스모크 테스트용)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="동시에 처리할 청크 수(LLM 추출 병렬, 쓰기는 직렬). 8GB VRAM 은 2~3 권장",
    )
    args = parser.parse_args()

    if args.qdrant:
        asyncio.run(
            ingest_qdrant(
                args.collection,
                args.qdrant_path,
                max_chunks=args.max_chunks,
                concurrency=args.concurrency,
            )
        )
        return

    if not args.input:
        logger.error("입력 경로가 필요합니다(또는 --qdrant 사용).")
        sys.exit(2)

    path = Path(args.input)
    if not path.exists():
        logger.error("입력 경로가 존재하지 않습니다: %s", path)
        sys.exit(2)

    files = iter_input_files(path)
    if not files:
        logger.error("처리할 pdf/txt 파일이 없습니다: %s", path)
        sys.exit(2)
    logger.info("대상 파일 %d개", len(files))

    asyncio.run(
        ingest_files(files, max_chunks=args.max_chunks, concurrency=args.concurrency)
    )


if __name__ == "__main__":
    main()

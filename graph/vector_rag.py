import argparse  # 커맨드라인 인자 파싱용
import logging   # 진행 상황 로깅용
import sys       # 프로그램 종료용
import time      # 실행 시간 측정용
from pathlib import Path  # 파일 경로 존재 여부 확인용

# vector_services에서 PDF 파서, 시맨틱 청커, 임베더 클래스를 임포트
from .vector_services import VehiclePDFParser, SemanticChunker, VehicleEmbedder

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorRAG:
    def __init__(self, model_name: str = "MiniLM"):
        """
        초기화 메서드. 추후 bge-m3 등 다른 임베딩 모델로 쉽게 교체할 수 있도록 설계
        """
        self.model_name = model_name
        logging.info(f"VectorRAG 초기화(Embedding Model: {self.model_name})")

    def run_chunk(self, pdf_path: str):
        """
        PDF 파싱 및 Semantic Chunking 수행
        """
        # 파일 존재 여부 검증 (방어적 코드)
        if not Path(pdf_path).exists():
            logging.error(f"지정된 PDF 파일을 찾을 수 없습니다: {pdf_path}")
            sys.exit(1)

        logging.info(f"문서 파싱 시작: {pdf_path}")
        t0 = time.time()
        documents = VehiclePDFParser().parse(pdf_path)
        logging.info(f"문서 파싱 완료 (소요시간: {time.time() - t0:.2f}초)")
        
        logging.info("Semantic Chunking 시작")
        t1 = time.time()

        chunker = SemanticChunker()
        chunks = chunker.chunk(documents)
        logging.info(f"Semantic Chunking 완료 (소요시간: {time.time() - t1:.2f}초)")

        # 토큰 통계 계산 및 로깅
        if chunks:
            token_counts = [len(c.page_content.split()) for c in chunks]
            avg_tokens = sum(token_counts) / len(token_counts)
            logging.info(f"생성된 청크 수: {len(chunks)}개, 평균 토큰 크기: {avg_tokens:.1f} (목표 512 내외)")
        else:
            logging.warning("생성된 청크가 없습니다.")

        return chunks, chunker

    def run_embed(self, pdf_path: str):
        """
        청킹 결과 임베딩 및 Vector Store 저장
        """
        # self를 통해 내부 메서드 호출 (스코프 수정)
        chunks, chunker = self.run_chunk(pdf_path)

        if not chunks:
            logging.warning("임베딩할 청크가 없어 종료합니다.")
            return

        logging.info("임베딩 및 Vector Store 저장 시작")
        t0 = time.time()
        embedder = VehicleEmbedder(embeddings=chunker._embeddings)
        embedder.embed_and_store(chunks)
        logging.info(f"임베딩 및 저장 완료 (소요시간: {time.time() - t0:.2f}초)")


def main():
 
    parser = argparse.ArgumentParser(description="Vector RAG 실행")
    parser.add_argument("pdf_path", help="PDF 파일 경로")
    parser.add_argument(
        "--step",
        choices=["chunk", "embed"],
        default="embed",
        help="실행 단계 선택 (기본: embed)",
    )
    # 모델 선택 인자 추가 (확장성 고려)
    parser.add_argument(
        "--model",
        default="MiniLM",
        help="사용할 임베딩 모델 (기본: MiniLM)",
    )
    args = parser.parse_args()

    # 인스턴스 생성 시 모델명 주입
    rag_pipeline = VectorRAG(model_name=args.model)

    if args.step == "chunk":
        chunks, _ = rag_pipeline.run_chunk(args.pdf_path)
        logging.info("=== 상위 5개 청크 샘플 ===")
        for i, doc in enumerate(chunks[:5]):
            logging.info(f"[Chunk {i+1}] {doc.page_content[:100]}...")
    elif args.step == "embed":
        rag_pipeline.run_embed(args.pdf_path)

if __name__ == "__main__":
    main()

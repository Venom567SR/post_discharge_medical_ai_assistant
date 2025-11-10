import argparse
from pathlib import Path
from typing import List
from pypdf import PdfReader
from loguru import logger

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_TOP_K,
    RAG_SCORE_THRESHOLD,
    VECTOR_STORE
)
from src.schemas import RetrievedChunk, Citation
from src.utils.chunking import chunk_text
from src.rag.embeddings import get_embedding_generator
from src.rag.vectorstore_qdrant import get_vector_store as get_qdrant_store
from src.rag.vectorstore_chroma import get_chroma_store
from src.utils.timing import Timer


class RAGRetriever:
    """RAG retrieval system for medical references."""
    
    def __init__(self, vector_store_type: str = VECTOR_STORE):
        """
        Initialize RAG retriever.
        
        Args:
            vector_store_type: 'qdrant' or 'chroma'
        """
        self.vector_store_type = vector_store_type
        self.embedding_generator = get_embedding_generator()
        
        if vector_store_type == "chroma":
            self.vector_store = get_chroma_store()
        else:
            self.vector_store = get_qdrant_store()
        
        logger.info(f"Initialized RAG retriever with {vector_store_type}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[dict]:
        """
        Extract text from PDF with page numbers.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            List of dicts with 'text' and 'page' keys
        """
        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text.strip():
                    pages.append({
                        "text": text,
                        "page": page_num
                    })
            
            logger.info(f"Extracted text from {len(pages)} pages in {pdf_path.name}")
            return pages
        
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def build_index(self, pdf_path: Path):
        """
        Build vector index from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Building index from {pdf_path}")
        
        # Step 1: Extract text from PDF
        with Timer("PDF text extraction"):
            pages = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Chunk the text
        all_chunks = []
        for page_data in pages:
            chunks = chunk_text(
                page_data["text"],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": pdf_path.name,
                    "page": page_data["page"]
                })
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Step 3: Generate embeddings
        with Timer("Generating embeddings"):
            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = self.embedding_generator.batch_embed(
                texts,
                batch_size=32,
                show_progress=True
            )
        
        # Step 4: Upsert to vector store
        ids = [f"{pdf_path.stem}_{i}" for i in range(len(all_chunks))]
        metadatas = [
            {"source": chunk["source"], "page": chunk["page"]}
            for chunk in all_chunks
        ]
        
        with Timer("Upserting to vector store"):
            self.vector_store.upsert(
                ids=ids,
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas
            )
        
        logger.info(f"Successfully built index with {len(all_chunks)} chunks")
        logger.info(f"Total vectors in store: {self.vector_store.count()}")
    
    def retrieve(
        self,
        query: str,
        k: int = RAG_TOP_K,
        score_threshold: float = RAG_SCORE_THRESHOLD
    ) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            k: Number of results to return
            score_threshold: Minimum similarity score
        
        Returns:
            List of retrieved chunks
        """
        if not query:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.embed_query(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k,
                score_threshold=score_threshold
            )
            
            logger.info(f"Retrieved {len(results)} chunks for query: {query[:100]}")
            return results
        
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return []
    
    def retrieve_with_citations(
        self,
        query: str,
        k: int = RAG_TOP_K,
        score_threshold: float = RAG_SCORE_THRESHOLD
    ) -> tuple[List[RetrievedChunk], List[Citation]]:
        """
        Retrieve chunks and format citations.
        
        Args:
            query: User query
            k: Number of results
            score_threshold: Minimum score
        
        Returns:
            Tuple of (chunks, citations)
        """
        chunks = self.retrieve(query, k, score_threshold)
        
        citations = [
            Citation(
                source_type="reference",
                reference_id=chunk.source,
                page=chunk.page,
                url=None,
                score=chunk.score,
                snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            )
            for chunk in chunks
        ]
        
        return chunks, citations
    
    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks as context for LLM.
        
        Args:
            chunks: Retrieved chunks
        
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found in references."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f" (page {chunk.page})" if chunk.page else ""
            context_parts.append(
                f"[Reference {i}{page_info}]:\n{chunk.text}\n"
            )
        
        return "\n".join(context_parts)


# Global singleton
_rag_retriever = None

def get_rag_retriever() -> RAGRetriever:
    """Get or create the global RAG retriever instance."""
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
    return _rag_retriever


def main():
    """CLI entry point for building RAG index."""
    parser = argparse.ArgumentParser(description="Build RAG index from PDF")
    parser.add_argument(
        "--build-index",
        type=str,
        metavar="PDF_PATH",
        help="Path to PDF file to index"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default=VECTOR_STORE,
        choices=["qdrant", "chroma"],
        help="Vector store to use (default: qdrant)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Test query after building index"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )
    
    args = parser.parse_args()
    
    if args.build_index:
        pdf_path = Path(args.build_index)
        retriever = RAGRetriever(vector_store_type=args.vector_store)
        retriever.build_index(pdf_path)
        
        print(f"\n✓ Successfully built index from {pdf_path.name}")
        print(f"  Vector store: {args.vector_store}")
        print(f"  Total chunks: {retriever.vector_store.count()}")
        
        # Test query if provided
        if args.query:
            print(f"\nTesting query: {args.query}")
            chunks = retriever.retrieve(args.query, k=args.top_k)
            print(f"Found {len(chunks)} results:")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n  Result {i} (score: {chunk.score:.3f}):")
                print(f"    Source: {chunk.source}, Page: {chunk.page}")
                print(f"    Text: {chunk.text[:150]}...")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

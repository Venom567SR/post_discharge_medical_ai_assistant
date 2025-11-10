from typing import List, Optional
import chromadb
from chromadb.config import Settings
from loguru import logger

from src.config import CHROMA_PATH
from src.schemas import RetrievedChunk
from src.utils.timing import Timer


class ChromaVectorStore:
    """ChromaDB vector store for document retrieval."""
    
    def __init__(
        self,
        collection_name: str = "medical_references",
        path: str = CHROMA_PATH
    ):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            collection_name: Name of the Chroma collection
            path: Path to store Chroma data
        """
        self.collection_name = collection_name
        self.path = path
        self.client = None
        self.collection = None
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.path,
                settings=Settings(anonymized_telemetry=False)
            )
            logger.info(f"Initialized ChromaDB client at {self.path}")
            self._get_or_create_collection()
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _get_or_create_collection(self):
        """Get or create the collection."""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Using collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            raise
    
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[dict]
    ):
        """
        Insert or update vectors in the collection.
        
        Args:
            ids: List of unique identifiers
            embeddings: List of embedding vectors
            texts: List of text chunks
            metadatas: List of metadata dictionaries
        """
        if not (len(ids) == len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("All input lists must have the same length")
        
        try:
            with Timer(f"Upserting {len(ids)} points to ChromaDB"):
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
            
            logger.info(f"Upserted {len(ids)} documents to {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to upsert to ChromaDB: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[dict] = None
    ) -> List[RetrievedChunk]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            score_threshold: Minimum similarity score (note: Chroma uses distance)
            filter_dict: Optional metadata filters
        
        Returns:
            List of RetrievedChunk objects
        """
        try:
            with Timer(f"Searching ChromaDB for top-{k} results"):
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=filter_dict if filter_dict else None
                )
            
            # ChromaDB returns results in a nested structure
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            # Convert distances to similarity scores (cosine: similarity = 1 - distance)
            chunks = []
            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Convert distance to similarity score
                score = 1.0 - distance
                
                # Apply score threshold if provided
                if score_threshold and score < score_threshold:
                    continue
                
                chunk = RetrievedChunk(
                    text=doc,
                    source=metadata.get("source", "unknown"),
                    page=metadata.get("page"),
                    score=score,
                    metadata={
                        k: v for k, v in metadata.items()
                        if k not in ["source", "page"]
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} results from ChromaDB")
            return chunks
        
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            return []
    
    def count(self) -> int:
        """Get the number of vectors in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            return {
                "name": self.collection_name,
                "vector_count": self.count(),
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


# Global singleton
_chroma_store = None

def get_chroma_store() -> ChromaVectorStore:
    """Get or create the global ChromaDB vector store instance."""
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaVectorStore()
    return _chroma_store

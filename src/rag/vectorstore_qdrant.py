import uuid
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from loguru import logger

from src.config import QDRANT_PATH, EMBEDDING_DIMENSION
from src.schemas import RetrievedChunk
from src.utils.timing import Timer


class QdrantVectorStore:
    """Qdrant vector store for document retrieval."""
    
    def __init__(
        self,
        collection_name: str = "medical_references",
        path: str = QDRANT_PATH,
        embedding_dim: int = EMBEDDING_DIMENSION
    ):
        """
        Initialize Qdrant client and collection.
        
        Args:
            collection_name: Name of the Qdrant collection
            path: Path to store Qdrant data
            embedding_dim: Dimension of embedding vectors
        """
        self.collection_name = collection_name
        self.path = path
        self.embedding_dim = embedding_dim
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize Qdrant client."""
        try:
            self.client = QdrantClient(path=self.path)
            logger.info(f"Initialized Qdrant client at {self.path}")
            self._ensure_collection_exists()
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
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
            ids: List of unique identifiers (strings will be converted to UUIDs)
            embeddings: List of embedding vectors
            texts: List of text chunks
            metadatas: List of metadata dictionaries
        """
        if not (len(ids) == len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("All input lists must have the same length")

        try:
            # Convert string IDs to UUIDs using UUID v5 (deterministic)
            # Use a namespace UUID for consistency
            namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace

            points = [
                PointStruct(
                    id=str(uuid.uuid5(namespace, id_)),
                    vector=embedding,
                    payload={
                        "text": text,
                        "original_id": id_,  # Store original ID for reference
                        **metadata
                    }
                )
                for id_, embedding, text, metadata in zip(ids, embeddings, texts, metadatas)
            ]

            with Timer(f"Upserting {len(points)} points to Qdrant"):
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

            logger.info(f"Upserted {len(points)} points to {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to upsert to Qdrant: {e}")
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
            score_threshold: Minimum similarity score
            filter_dict: Optional metadata filters
        
        Returns:
            List of RetrievedChunk objects
        """
        try:
            # Build filter if provided
            query_filter = None
            if filter_dict:
                conditions = [
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in filter_dict.items()
                ]
                if conditions:
                    query_filter = Filter(must=conditions)
            
            with Timer(f"Searching Qdrant for top-{k} results"):
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=k,
                    query_filter=query_filter,
                    score_threshold=score_threshold
                )
            
            # Convert to RetrievedChunk objects
            chunks = []
            for result in results:
                chunk = RetrievedChunk(
                    text=result.payload.get("text", ""),
                    source=result.payload.get("source", "unknown"),
                    page=result.payload.get("page"),
                    score=result.score,
                    metadata={
                        k: v for k, v in result.payload.items()
                        if k not in ["text", "source", "page"]
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Found {len(chunks)} results from Qdrant")
            return chunks
        
        except Exception as e:
            logger.error(f"Failed to search Qdrant: {e}")
            return []
    
    def count(self) -> int:
        """Get the number of vectors in the collection."""
        try:
            result = self.client.count(collection_name=self.collection_name)
            return result.count
        except Exception as e:
            logger.error(f"Failed to count vectors: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vector_count": self.count(),
                "embedding_dim": self.embedding_dim
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


# Global singleton
_vector_store = None

def get_vector_store() -> QdrantVectorStore:
    """Get or create the global Qdrant vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore()
    return _vector_store

from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION
from src.utils.timing import Timer


class EmbeddingGenerator:
    """Wrapper for sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            with Timer("Loading embedding model", log=True):
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded embedding model: {self.model_name}")
                logger.info(f"Embedding dimension: {self.get_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of strings to embed
        
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []
        
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            with Timer(f"Embedding {len(texts)} texts"):
                embeddings = self.model.encode(
                    texts,
                    show_progress_bar=False,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector as list of floats
        """
        if not query:
            raise ValueError("Query cannot be empty")
        
        embeddings = self.embed_texts([query])
        return embeddings[0] if embeddings else []
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        if self.model is None:
            return EMBEDDING_DIMENSION
        return self.model.get_sentence_embedding_dimension()
    
    def batch_embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings in batches for large datasets.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            show_progress: Whether to show progress bar
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            logger.info(f"Batch embedding {len(texts)} texts with batch_size={batch_size}")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to batch embed: {e}")
            raise


# Global singleton instance
_embedding_generator = None

def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator

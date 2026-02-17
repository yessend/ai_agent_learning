from qdrant_client import AsyncQdrantClient, QdrantClient

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.core.config import Config
from helpers.logger import logger


class RagIngestion:
    
    def __init__(self):
        """Initialize the documents ingestion part of the RAG and form a knowledge base."""
        self.embedding = HuggingFaceEmbedding(
            model_name=Config.EMBEDDING_MODEL
        )
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT,
            timeout=300
        )
        self.qdrant_aclient = AsyncQdrantClient(
            url=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT,
            timeout=300
        )
        
    
    def ingest(self):
        """Loads the Qdrant-based index for querying"""
        try:
            
            vector_store = QdrantVectorStore(
                aclient=self.qdrant_aclient,
                client=self.qdrant_client, 
                collection_name=Config.QDRANT_COLLECTION_NAME,
                enable_hybrid=True,
                fastembed_sparse_model="Qdrant/bm25",
                dense_vector_name="dense_embed_vector",
                sparse_vector_name="sparse_keyword_vector"
            )

            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embedding,
                use_async=True
            )

            hybrid_retriever = vector_index.as_retriever(
                vector_store_query_mode="hybrid", 
                similarity_top_k=20,  
                alpha=0.5
            )
            
            logger.info("Loaded index with robust selector successfully.")
            return hybrid_retriever
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return None
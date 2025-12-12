from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from core.config.config import Config

from llama_index.core import VectorStoreIndex
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMMultiSelector
from core.config.constants import RagConstants
from core.helpers.qdrant_setup import DualSchemaQdrantVectorStore

from qdrant_client import QdrantClient


class RagSystem:
    """
    Rag pipeline implemented as a workflow.
    """
    def __init__(
        self,
        qdrant_client: QdrantClient | None,
        router_llm: GoogleGenAI | None,
        embed_model: HuggingFaceEmbedding | None
    ):
        self.qdrant_client = qdrant_client
        self.router_llm = router_llm
        self.embed_model = embed_model
        
        self.router_retriever = self._get_router()

        
    def _get_router(self) -> RouterRetriever | None:
        """
        Args:
            self: RagSystem class instance itself
        
        The given method returns the RouterRetriever object, which using the LLMMultiSelector selects
        the most appropriate retrievers from the list of RetrieverTool objects (which are just wrapper around
        retrievers from index with added desrcription for LLM selection). Number of selected retrievers is
        ROUTER_RETRIEVER_MAX_OUTPUTS from the Config file, while each retrievers selects SIMILARITY_TOP_K (from
        Config as well) nodes from each index (which is created from distinct Qdrant collection).
        
        Returns:
            RouterRetriever | None: explained above
        """
        self._retriever_tools = []
        
        for collection_name, colleciton_description in RagConstants.COLLECTIONS.items():
            
            collection_store = DualSchemaQdrantVectorStore(
                collection_name = collection_name,
                client = self.qdrant_client,
                aclient = None
            )
            
            collection_index = VectorStoreIndex.from_vector_store(
                vector_store = collection_store,
                embed_model = self.embed_model
            )
            
            collection_retriever = collection_index.as_retriever(similarity_top_k = Config.SIMILARITY_TOP_K)
            
            collection_retriever_tool = RetrieverTool.from_defaults(
                retriever = collection_retriever,
                description = colleciton_description
            )
            
            self._retriever_tools.append(collection_retriever_tool)
            
        router = RouterRetriever(
            selector = LLMMultiSelector.from_defaults(
                prompt_template_str = RagConstants.LLM_MULTI_SELECTOR_PROMPT,
                # Maximum number of retrievers to retain - each retriever retrieves nodes from each corresponding colleciton
                max_outputs = Config.ROUTER_RETRIEVER_MAX_OUTPUTS,
                llm = self.router_llm
            ),
            llm = self.router_llm,
            retriever_tools = self._retriever_tools
        )
        return router
    
    # Just to check the relevance prompt
    def _relevance_filter(self, retrieved_nodes, query: str):
        """
        Checks if retrieved nodes from the index (buiilt on top of vector database)
        contains relevant information to the user's prompt.
        """
        
        if not retrieved_nodes:
            return False
        
        relevant_nodes = [node.text for node in retrieved_nodes]
        
        context = "\n".join(relevant_nodes)
        
        relevance_prompt = f"""
        Question: {query}
        
        Retrieved context:
        {context}
        
        Does the retrieved context contain information that can help answer the question? 
        Answer with only 'YES' or 'NO'.
        """
        return relevance_prompt
            
    
    
    
    
    """
    def pipeline(self, query) -> None:
        
        # 1. Classify intent
        
        # 2. Use LLM to answer small talk
        
        # 3. If not small talk, use router retriever
        retrieved_nodes = self.router_retriever.retrieve(query)
        
        # 4. Check relevance:
        is_relevant = _relevance_filter(retrieved_nodes)
    """
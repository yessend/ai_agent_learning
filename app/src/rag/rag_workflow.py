# Implement a custom RAG System from scratch as subclass of Workflow 
# allowing multiple users have a chat history.
from app.core.config import Config
from app.core.constants import RagConstants
from app.src.rag.custom_chat_engine import GeminiChatEngine
from app.src.rag.rag_ingestion import RagIngestion
from app.src.rag.rag_events import RagGenerateEvent
from app.src.rag.redis_client import RedisClient

from helpers.logger import logger

from google.genai import Client
from uuid import uuid5, NAMESPACE_DNS

from llama_index.core.workflow import (
    Workflow,
    Context,
    StartEvent,
    StopEvent,
    step
)


class RagChatWorkflow(Workflow):
    # This is the whole RAG system implemented as a Workflow
    
    def __init__(self):
        super().__init__()
        self.redis_async_client = RedisClient().redis_pool_init()
        self.hybrid_retriever = RagIngestion().ingest()
        self.gemini_client = Client(api_key=Config.GOOGLE_API_KEY)

    @step
    async def _retrieve_nodes(self, ctx: Context, ev: StartEvent) -> RagGenerateEvent | StopEvent:
        
        user_query = ev.get("user_query")
        user_name = ev.get("user_name")
        user_id = ev.get("user_id")
        
        await ctx.store.set("user_query", user_query)
        await ctx.store.set("user_name", user_name)
        await ctx.store.set("user_id", user_id)

        try:            
            # Retrieve from your RAG system
            if self.hybrid_retriever:
                retrieved_nodes = await self.hybrid_retriever.aretrieve(user_query)
                return RagGenerateEvent(retrieved_nodes = retrieved_nodes)
            else:
                logger.warning("No router_retriever was initialized for this RAG workflow...")
                return RagGenerateEvent(retrieved_nodes = None)
        except ValueError as e:
            logger.warning(f"Error during nodes retrieval, internal RAG failed for user {user_id}: {e}. Try to use Gemini grounding...")
            return RagGenerateEvent(retrieved_nodes = None)
        except Exception as e:
            logger.error(f"Fatal error during nodes retrieval, internal RAG failed for user {user_id}: {e}")
            return StopEvent(result = "There was an error in the service, try later...")


    @step
    async def _rag_synthesize(self, ctx: Context, ev: RagGenerateEvent) -> StopEvent:
        
        """Generate response using Gemini and use Google Search grounding if retrieved nodes are irrelevant (WITH MANUAL TRACING)"""
        
        retrieved_nodes = ev.retrieved_nodes

        user_query = await ctx.store.get("user_query")
        user_name = await ctx.store.get("user_name")
        user_id = await ctx.store.get("user_id")
         
        logger.info(f"Synthesizing an answer for user {user_id} either using RAG or grounding...")
        
        try:
            context = ""
            if retrieved_nodes: 
                context = "\n".join([node.text if node.text != "None" else str(node.metadata) for node in retrieved_nodes])
            
            chat_engine = GeminiChatEngine(
                llm_client=self.gemini_client, 
                system_prompt=RagConstants.SYSTEM_PROMPT_WORKFLOW,
                redis_async_client=self.redis_async_client,
                redis_store_key=str(uuid5(NAMESPACE_DNS, str(user_id))),
                history_fetch_limit=Config.CHAT_HISTORY_FETCH_LIMIT,
                history_token_limit=Config.CHAT_HISTORY_TOKEN_LIMIT           
            )

            response = await chat_engine.achat(user_query, user_name, "ENGLISH", context)
            
            supports = response.candidates[0].grounding_metadata.grounding_supports
            response_text = response.text

            if supports:
                logger.info("Gemini used grounding for a knowledge question, adding disclaimer...")
                disclaimer = "🔍 Information found through external sources:\n\n"
                response_text = disclaimer + response_text
                logger.info(f"The sources for the answer{supports}")
                logger.info(f"Successfully processed grounded query for user {user_id}")
            else:
                logger.info("Gemini generated answer based on provided context or memory, suppressing disclaimer...")

            return StopEvent(result = response_text)
    
        except Exception as e:
            response_text = ""
            if ev.rag_gen_failed:
                response_text = "Internal knolwedge search and external knowledge failed..."
                logger.error(f"Error answering user {user_id} question: {e}")
            else:
                response_text = "The bot could not answer the question..."
                logger.error(f"Both internal knowledge and Gemini grounding failed for user {user_id}: {e}")
            return StopEvent(result = response_text)
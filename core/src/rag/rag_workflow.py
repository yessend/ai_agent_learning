# Implement a custom RAG System from scratch as subclass of Workflow 
# allowing multiple users have a chat history.
from core.config.config import Config
from core.config.constants import RagConstants
from core.config.llm_setup import LLMsetups
from core.src.rag.custom_chat_engine import CustomSimpleChatEngine
from core.src.rag.rag_ingestion import RagIngestion
from core.config.constants import RagConstants
from core.src.rag.rag_events import RetrievalRelevantEvent

from helpers.logger import logger
from helpers.json_extractor import extract_json_array

import redis
import redis.asyncio as async_redis

from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
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
        self.token_counter = TokenCountingHandler()
        
        if Settings.callback_manager:
            Settings.callback_manager.add_handler(self.token_counter)
        else:
            Settings.callback_manager = CallbackManager([self.token_counter])
        
        self.router_llm = LLMsetups.ROUTER_LLM
        self.chat_llm = LLMsetups.CHAT_LLM
        
        self.router_llm.callback_manager = Settings.callback_manager
        self.chat_llm.callback_manager = Settings.callback_manager
        
        self.redis_chat_store = self.redis_chat_store_init()
        self.router_retriever = RagIngestion().ingest()
    
    # --------------------------------------------------------------------------------
    # Helper method to initialize the Redis Async client and return the RedisChatStore
    def redis_chat_store_init(self) -> RedisChatStore:

        async_pool = async_redis.BlockingConnectionPool.from_url(
            Config.REDIS_URL, 
            max_connections = Config.REDIS_MAX_CONNECTIONS,
            timeout = Config.REDIS_TIMEOUT, 
            decode_responses = False
        )

        custom_async_client = async_redis.Redis(connection_pool = async_pool)

        # For llama_index
        sync_pool = redis.ConnectionPool.from_url(
            Config.REDIS_URL,
            max_connections = Config.REDIS_MAX_CONNECTIONS,
            decode_responses = False
        )
        custom_sync_client = redis.Redis(connection_pool = sync_pool)

        # 3. Initialize Store with BOTH
        REDIS_STORE = RedisChatStore(
            redis_client = custom_sync_client,   
            aredis_client = custom_async_client, 
            ttl = Config.REDIS_TTL
        )

        return REDIS_STORE
    # --------------------------------------------------------------------------------

    
    @step
    async def _is_retrieval_relevant(self, ctx: Context, ev: StartEvent) -> RetrievalRelevantEvent | None:
        # Check the relevance of the nodes retrieved from router retriever.
        # We will return either False value if no retrieval is irrelevant or the retrieved
        # context otherwise
        user_query = ev.get("user_query")
        user_name = ev.get("user_name")
        user_id = ev.get("user_id")
        
        if not user_query or not self.router_retriever or not user_name or not user_id:
            logger.warning("Relevancy check cannot be performed, missing arguments in the Start Event.")
            return None
        
        await ctx.store.set("user_query", user_query)
        if user_name:
            await ctx.store.set("user_name", user_name)
        if user_id:
            await ctx.store.set("user_id", user_id)
        
        
        try:
            retrieved_nodes = await self.router_retriever.aretrieve(user_query)
        except ValueError as e:
            logger.warning(f"{e}: knowledge base does not contain relevant info; no nodes were retrieved")
            return RetrievalRelevantEvent(context = False)
        
        # Create retrieved_nodes str for LLM to easier make a choice
        retrieved_nodes_str = ""
        for i in range(len(retrieved_nodes)):
            if i == len(retrieved_nodes) - 1: 
                str_to_add = "\"" + retrieved_nodes[i].id_ + "\": \"\"\"" + retrieved_nodes[i].text + "\"\"\""
            else:
                str_to_add = "\"" + retrieved_nodes[i].id_ + "\": \"\"\""  + retrieved_nodes[i].text + "\"\"\"\n"
            retrieved_nodes_str += str_to_add

        # Check the relevance using the LLM call
        relevance_check_prompt = RagConstants.LLM_RELEVANCE_CHECK_PROMPT.format(
            question = user_query,
            context = retrieved_nodes_str
        )
        # Make an LLM call to get relevant nodes
        response_relevance = await self.router_llm.acomplete(relevance_check_prompt)
        
        # Now, we derive the relevant context from the LLM response and construct the final
        # context string that we will later use in the final LLM call to generate an answer to the user query
        retrieved_nodes_dict = dict([(node.id_, node.text) for node in retrieved_nodes])

        relevant_node_ids = extract_json_array(response_relevance.text.strip())
        
        if not relevant_node_ids:
            logger.warning("Among retrieved nodes, no nodes contain relevant information to the user's query.")
            return RetrievalRelevantEvent(context = False)
        
        # if not relevant_node_ids:
        #     logger.warning("Retrieved nodes do not containt a relevant info.")
        # else:
        #     pass

        context = "\n\n".join(
            [retrieved_nodes_dict[node_id] for node_id in relevant_node_ids]
        )

        return RetrievalRelevantEvent(context = context)
    
    
    @step
    async def _synthesize(self, ctx: Context, ev: RetrievalRelevantEvent) -> StopEvent | None:
        # Here we produce the final response to the user's query
        user_query = await ctx.store.get("user_query", default = None)
        user_name = await ctx.store.get("user_name", default = None)
        user_id = await ctx.store.get("user_id", default = None)
        
        context = ev.context
        
        if not context:
            logger.warning("No context is provided.")

        memory = ChatMemoryBuffer.from_defaults(
            token_limit = Config.CHAT_MEMORY_TOKEN_LIMIT,
            chat_store = self.redis_chat_store,            
            chat_store_key = f"user_{user_id}",
            llm = self.chat_llm             
        )

        chat_engine = CustomSimpleChatEngine.from_defaults(
            llm = self.chat_llm,           
            memory = memory,               
            system_prompt = RagConstants.SYSTEM_PROMPT_WORKFLOW  
        )
        
        response = await chat_engine.achat(user_query, user_name, context)
        return StopEvent(result = response.response)
# Implement a custom RAG System from scratch as subclass of Workflow 
# allowing multiple users have a chat history.
from core.config.constants import RagConstants
from core.config.llm_setup import LLMsetups
from core.src.chat_engine import CustomContextChatEngine
from core.src.rag_ingestion import RagIngestionRetriever
from core.helpers.logger import logger
from core.config.config import Config
from core.helpers.json_extractor import extract_json_array

from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
from llama_index.core.memory import Memory
from llama_index.core.workflow import (
    Workflow,
    Context,
    Event,
    StartEvent,
    StopEvent,
    step
)


# These are the custom classes to perform RAG and synthesis of an answer:
class RetrievalRelevantEvent(Event):
    context: str | bool


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
        
        self.chat_engines_cached: dict[str, CustomContextChatEngine] = {}
        self.custom_retriever = RagIngestionRetriever()
        
    # ------------------------------------------------------------------------------------
    # Helper methods
    def get_or_create_chat_engine(self, user_id: str) -> CustomContextChatEngine:
        # Here we either get cached chat_engine (SimpleChatEngine object) for a specific user 
        # or we create a new chat_engine for the user if he hadn't one before.
        if user_id in self.chat_engines_cached:
            logger.info(f"Using cached engine for the user {user_id}")
            return self.chat_engines_cached[user_id]
        
        logger.info(f"Creating a new chat engine for user {user_id}")
        
        context_prompt = (
            "Use the context information below to answer user's question."
            "\n--------------------\n"
            "{context_str}"
            "\n--------------------\n"
        )
        
        memory = Memory.from_defaults(
            session_id = "Test_session_id", # change it later on a correct session_id
            token_limit = Config.CHAT_MEMORY_TOKEN_LIMIT,
            chat_history_token_ratio = Config.CHAT_HISTORY_TOKEN_RATIO
        )
        
        chat_engine = CustomContextChatEngine.from_defaults(
            system_prompt=RagConstants.SYSTEM_PROMPT_WORKFLOW,
            context_template=context_prompt,
            retriever=self.custom_retriever,
            llm=self.chat_llm,
            memory=memory
        )
        self.chat_engines_cached[user_id] = chat_engine
        return chat_engine
    # ------------------------------------------------------------------------------------
    
    @step
    async def _synthesize(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        # Here we produce the final response to the user's query
        user_query = ev.get("user_query")
        user_name = ev.get("user_name")
        user_id = ev.get("user_id")
        
        if not user_query or not self.custom_retriever or not user_name or not user_id:
            logger.warning("Relevancy check cannot be performed, missing arguments in the Start Event.")
            return None
        
        chat_engine = self.get_or_create_chat_engine(user_id)
        
        response = await chat_engine.achat(user_query)
        if response is None:
            return StopEvent(result = "Response is None, retriever is irrelevant")
        else:
            return StopEvent(result = response.response)
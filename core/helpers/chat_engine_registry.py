from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.core.memory import Memory

from core.config.config import Config
from core.config.constants import RagConstants
from core.helpers.logger import logger

class ChatEngineRegistry:
    # Create a class that we will use as a dependency for the Workflow.
    # We will then inject this dependency upon creating a RagWorkflow class
    def __init__(self, chat_llm: GoogleGenAI | None):
        self.chat_llm = chat_llm
        self.chat_engines_cached: dict[str, SimpleChatEngine] = {}
    
    def get_or_create_chat_engine(self, user_id: str) -> SimpleChatEngine:
        # Here we either get cached chat_engine (SimpleChatEngine object) for a specific user 
        # or we create a new chat_engine for the user if he hadn't one before.
        if user_id in self.chat_engines_cached:
            logger.info(f"Using cached engine for the user {user_id}")
            return self.chat_engines_cached[user_id]
        
        logger.info(f"Creating a new chat engine for user {user_id}")
        
        memory = Memory.from_defaults(
            session_id = "Test_session_id", # change it later on a correct session_id
            token_limit = Config.CHAT_MEMORY_TOKEN_LIMIT,
            chat_history_token_ratio = 1.0
        )
        
        chat_engine = SimpleChatEngine.from_defaults(
            system_prompt = RagConstants.SYSTEM_PROMPT_WORKFLOW,
            llm = self.chat_llm,
            memory = memory
        )
        self.chat_engines_cached[user_id] = chat_engine
        return chat_engine
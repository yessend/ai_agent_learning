from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.core.memory import Memory
from llama_index.core.agent import AgentChatResponse
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import trace_method

from core.config.config import Config
from core.config.constants import RagConstants
from core.helpers.logger import logger

from typing import Optional, List

class CustomSimpleChatEngine(SimpleChatEngine):
    
    @trace_method("chat")
    async def achat(
        self, 
        message: str, 
        context: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            await self._memory.aset(chat_history)
        await self._memory.aput(ChatMessage(content=message, role="user"))

        if hasattr(self._memory, "tokenizer_fn"):
            initial_token_count = len(
                self._memory.tokenizer_fn(
                    " ".join(
                        [
                            (m.content or "")
                            for m in self._prefix_messages
                            if isinstance(m.content, str)
                        ]
                    )
                )
            )
        else:
            initial_token_count = 0
        
        context_prompt = (
            "Use the context information below to answer user's question.\n"
            f"<{context}>"
        )
        
        context_wrapped = ChatMessage(content=context_prompt, role="user")

        all_messages = self._prefix_messages + (
            await self._memory.aget(initial_token_count=initial_token_count)
        ) + [context_wrapped]

        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        await self._memory.aput(ai_message)

        return AgentChatResponse(response=str(chat_response.message.content))
    
    @property
    def memory(self) -> Memory:
        return self._memory


class ChatEngineRegistry:
    # Create a class that we will use as a dependency for the Workflow.
    # We will then inject this dependency upon creating a RagWorkflow class
    def __init__(self, chat_llm: GoogleGenAI | None):
        self.chat_llm = chat_llm
        self.chat_engines_cached: dict[str, CustomSimpleChatEngine] = {}
    
    def get_or_create_chat_engine(self, user_id: str) -> CustomSimpleChatEngine:
        # Here we either get cached chat_engine (SimpleChatEngine object) for a specific user 
        # or we create a new chat_engine for the user if he hadn't one before.
        if user_id in self.chat_engines_cached:
            logger.info(f"Using cached engine for the user {user_id}")
            return self.chat_engines_cached[user_id]
        
        logger.info(f"Creating a new chat engine for user {user_id}")
        
        memory = Memory.from_defaults(
            # session_id = "Test_session_id", # change it later on a correct session_id
            token_limit = Config.CHAT_MEMORY_TOKEN_LIMIT,
            chat_history_token_ratio = Config.CHAT_HISTORY_TOKEN_RATIO
        )
        
        chat_engine = CustomSimpleChatEngine.from_defaults(
            system_prompt = RagConstants.SYSTEM_PROMPT_WORKFLOW,
            llm = self.chat_llm,
            memory = memory
        )
        self.chat_engines_cached[user_id] = chat_engine
        return chat_engine
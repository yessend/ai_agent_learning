from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.storage.chat_store.redis import RedisChatStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.settings import Settings

from llama_index.core.utils import get_tokenizer
from llama_index.core.callbacks import trace_method

import json
from typing import List


class CustomChatEngine():
    
    def __init__(
        self,
        llm: GoogleGenAI | None,
        system_prompt: str,
        redis_store: RedisChatStore, 
        chat_store_key: str,
        token_limit: int,
        fetch_limit: int,
        tokenizer_fn = None,
    ):
        self._llm = llm
        self._prefix_messages = [ChatMessage(content = system_prompt, role = "system")]

        self.redis_store = redis_store
        self.chat_store_key = chat_store_key
        
        self.tokenizer_fn = tokenizer_fn or get_tokenizer()

        self.token_limit = token_limit
        self.fetch_limit = fetch_limit
        
        self.callback_manager = llm.callback_manager
    
    async def _get_history_safe(
        self,
        initial_token_count: int = 0
    ) -> List[ChatMessage]:
        """Get chat history."""
        
        # Not the correct way to go about it, check it later
        items = await self.redis_store._aredis_client.lrange(self.chat_store_key, -self.fetch_limit, -1)
        if len(items) == 0:
            return []

        items_json = [json.loads(m.decode("utf-8")) for m in items]
        chat_history = [ChatMessage.model_validate(d) for d in items_json]

        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")

        message_count = len(chat_history)

        cur_messages = chat_history[-message_count:]
        token_count = await self._token_count_for_messages(cur_messages) + initial_token_count

        while token_count > self.token_limit and message_count > 1:
            message_count -= 1
            while chat_history[-message_count].role in (
                MessageRole.TOOL,
                MessageRole.ASSISTANT,
            ):
                message_count -= 1

            cur_messages = chat_history[-message_count:]
            token_count = (
                await self._token_count_for_messages(cur_messages) + initial_token_count
            )

        # catch one message longer than token limit
        if token_count > self.token_limit or message_count <= 0:
            return []

        return chat_history[-message_count:]
    

    async def _token_count_for_messages(self, messages: List[ChatMessage]) -> int:
        if len(messages) <= 0:
            return 0

        msg_str = " ".join(str(m.content) for m in messages)
        return len(self.tokenizer_fn(msg_str))
    
    
    @trace_method("chat")
    async def achat(
        self, 
        message: str, 
        user_name: str,
        context: str
    ) -> AgentChatResponse:
        
        initial_token_count = len(
            self.tokenizer_fn(
                " ".join(
                    [
                        (m.content or "")
                        for m in self._prefix_messages
                        if isinstance(m.content, str)
                    ]
                )
            )
        )
        
        query_full = f"""
            User's name: {user_name}
            Question: {message}

            {f"Use the context information below to answer user's question.\n<context>\n{context}" if context else ""}
        """
        
        query_wrapped = ChatMessage(content=query_full, role="user")

        all_messages = self._prefix_messages + (
            await self._get_history_safe(initial_token_count=initial_token_count)
        ) + [query_wrapped]
        
        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        
        await self.redis_store.async_add_message(self.chat_store_key, ChatMessage(content=message, role="user"))
        await self.redis_store.async_add_message(self.chat_store_key, ai_message)

        return AgentChatResponse(response=str(chat_response.message.content))
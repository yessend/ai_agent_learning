from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.core.memory import Memory
from llama_index.core.agent import AgentChatResponse
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import trace_method

from typing import Optional, List


class CustomSimpleChatEngine(SimpleChatEngine):
    
    @trace_method("chat")
    async def achat(
        self, 
        message: str, 
        user_name: str,
        context: str,
        chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        if chat_history is not None:
            await self._memory.aset(chat_history)

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
        
        query_full = f"""
            User's name: {user_name}
            Question: {message}

            {f"Use the context information below to answer user's question.\n<context>\n{context}" if context else ""}
        """
        
        query_wrapped = ChatMessage(content=query_full, role="user")

        all_messages = self._prefix_messages + (
            await self._memory.aget(initial_token_count=initial_token_count)
        ) + [query_wrapped]
        
        chat_response = await self._llm.achat(all_messages)
        ai_message = chat_response.message
        
        await self._memory.aput(ChatMessage(content=message, role="user"))
        await self._memory.aput(ai_message)

        return AgentChatResponse(response=str(chat_response.message.content))
    
    @property
    def memory(self) -> Memory:
        return self._memory
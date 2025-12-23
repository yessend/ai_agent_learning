from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.chat_engine.types import ToolOutput
from llama_index.core.agent import AgentChatResponse
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import trace_method

from typing import Optional, List


class CustomContextChatEngine(ContextChatEngine):
    
    async def _aget_nodes(self, message: str) -> list[NodeWithScore] | None:
        """Generate context information from a message."""
        nodes = await self._retriever.aretrieve(message)
        if not nodes:
            return []
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        return nodes
    
    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> AgentChatResponse | None:
        if chat_history is not None:
            await self._memory.aset(chat_history)

        # get nodes and postprocess them
        nodes = await self._aget_nodes(message)
        if not nodes:
            return None
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        # Get the response synthesizer with dynamic prompts
        chat_history = await self._memory.aget(
            input=message,
        )
        synthesizer = self._get_response_synthesizer(chat_history)

        response = await synthesizer.asynthesize(message, nodes)
        user_message = ChatMessage(content=str(message), role=MessageRole.USER)
        ai_message = ChatMessage(content=str(response), role=MessageRole.ASSISTANT)

        await self._memory.aput(user_message)
        await self._memory.aput(ai_message)

        return AgentChatResponse(
            response=str(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
        )
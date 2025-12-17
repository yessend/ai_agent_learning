# Implement a custom RAG System from scratch as subclass of Workflow 
# allowing multiple users have a chat history.
from core.config.constants import RagConstants
from core.config.llm_setup import LLMsetups
from core.src.chat_engine_registry import ChatEngineRegistry
from core.src.rag_ingestion import RagIngestion
from core.helpers.logger import logger
from core.helpers.json_extractor import extract_json_array

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
        self.chat_engine_registry = ChatEngineRegistry(chat_llm = LLMsetups.CHAT_LLM)
        self.router_llm = LLMsetups.ROUTER_LLM
        self.chat_llm = LLMsetups.CHAT_LLM
        self.router_retriever = RagIngestion().ingest()
    
    
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
        prompt = ""
        
        if not context:
            logger.warning("No context is provided.")
            prompt = f"Name of the user: {user_name}; question: {user_query}"
        else:
            prompt = RagConstants.SYSTEM_PROMPT_WORKFLOW + f"""
                    User's name: {user_name}
                    User's question: {user_query}
                    Context: {context}
                    """
        
        chat_engine = self.chat_engine_registry.get_or_create_chat_engine(user_id)
        
        response = await chat_engine.achat(prompt)
        return StopEvent(result = response.response)
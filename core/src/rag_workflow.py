# Implement a custom RAG System from scratch as subclass of Workflow 
# allowing multiple users have a chat history.
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMMultiSelector

from core.config.config import Config
from core.config.constants import RagConstants
from core.helpers.chat_engine_registry import ChatEngineRegistry
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



# These are the custom classes to perform ingestion of documents:
class RetrieverToolsEvent(Event):
    retriever_tools: list[RetrieverTool]


class RagIngestionWorkflow(Workflow):
    # This is the RAG workflow to ingest the documents and form the knowledge base

    def __init__(
        self, 
        router_llm: GoogleGenAI | None,
        embed_model: HuggingFaceEmbedding | None
    ):
        super().__init__()
        self.router_llm = router_llm
        self.embed_model = embed_model


    @step
    async def _ingest(self, ev: StartEvent) -> RetrieverToolsEvent | None:
        # Initialize the retriever_tools list to create a list of RetrieverTool objects that we will later
        # pass into the LLMMultiSelector for selecting an appropriate retriever
        docs_path = ev.get("docs_path")
        collections = ev.get("collections")
        
        if not docs_path or not collections:
            logger.warning("Ingestion method cannot be performed, missing arguments in the Start Event.")
            return None
        
        retriever_tools = []

        # I manually wrote a dictionary for each course and its decription inside the previously loaded JSON file. 
        # 'collection_name' matches the name of the course folder inside the documents folder
        for collection_name, collection_description in collections.items():

            collection_path = docs_path + '/' + collection_name

            # 1) Read documents and create list of 'Document' objects, that has id_, metadata, text attributes.
            #    Document class (generic container for any data source) is a subclass of the TextNode class
            collection_documents = SimpleDirectoryReader(input_dir = collection_path).load_data()

            # 2) Read each of this document objects and create index from it
            #    Document objects are parsed into Node objects that have different attributes such as text, embeddings, metadata, relationships.
            #    Document objects are split into multiple nodes (relationships between these nodes are recorded in Node objects as attributes).
            collection_index = VectorStoreIndex.from_documents(
                documents = collection_documents,
                embed_model = self.embed_model,
                show_progress = True
            )

            # 3) Then we create a retriever from each of those indices that were built on top of those collections of Document objects
            #    To do it, we just call the as_retriever method of the VectorStoreIndex object
            #    We also indicate the similarity_top
            collection_retriever = collection_index.as_retriever(similarity_top_k = Config.SIMILARITY_TOP_K)

            # 4) We wrap those collection retrievers inside the RetrieverTool so that the MultiSelector will be able to select an
            #    appropriate retriever based on its decription
            collection_retriever_tool = RetrieverTool.from_defaults(
                retriever = collection_retriever,
                description = collection_description
            )

            # 5) Append created RetrieverTool for each collection to the list initialized before this loop
            retriever_tools.append(collection_retriever_tool)
        
        return RetrieverToolsEvent(retriever_tools = retriever_tools)
    
    
    @step
    async def _get_router_retriever(self, ev: RetrieverToolsEvent) -> StopEvent | None:
        # Create a router from that list of RetrieverTool objects using an LLMMultiSelector for selecting relevant retrievers 
        # based on a prompt
        router = RouterRetriever(
            selector = LLMMultiSelector.from_defaults(
                prompt_template_str = RagConstants.LLM_MULTI_SELECTOR_PROMPT,
                # Maximum number of retrievers to retain - each retriever retrieves nodes from each corresponding colleciton
                max_outputs = Config.ROUTER_RETRIEVER_MAX_OUTPUTS,
                llm = self.router_llm
            ),
            llm = self.router_llm,
            retriever_tools = ev.retriever_tools
        )
        return StopEvent(result = router)



# These are the custom classes to perform RAG and synthesis of an answer:
class RetrievalRelevantEvent(Event):
    context: str | bool


class RagChatWorkflow(Workflow):
    # This is the whole RAG system implemented as a Workflow
    
    def __init__(
        self, 
        chat_engine_registry: ChatEngineRegistry,
        router_llm: GoogleGenAI | None,
        chat_llm: GoogleGenAI | None,
    ):
        super().__init__()
        self.chat_engine_registry = chat_engine_registry
        self.router_llm = router_llm
        self.chat_llm = chat_llm
    
    
    @step
    async def _is_retrieval_relevant(self, ctx: Context, ev: StartEvent) -> RetrievalRelevantEvent | None:
        # Check the relevance of the nodes retrieved from router retriever.
        # We will return either False value if no retrieval is irrelevant or the retrieved
        # context otherwise
        router_retriever = ev.get("router_retriever")
        user_query = ev.get("user_query")
        user_name = ev.get("user_name")
        user_id = ev.get("user_id")
        
        if not user_query or not router_retriever or not user_name or not user_id:
            logger.warning("Relevancy check cannot be performed, missing arguments in the Start Event.")
            return None
        
        await ctx.store.set("user_query", user_query)
        if user_name:
            await ctx.store.set("user_name", user_name)
        if user_id:
            await ctx.store.set("user_id", user_id)
        
        
        try:
            retrieved_nodes = await router_retriever.aretrieve(user_query)
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
# Implement a custom RAG System from scratch as a class 
# allowing multiple users have a chat history.
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMMultiSelector

from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.core.memory import Memory

from core.config.config import Config
from core.config.constants import RagConstants
from core.helpers.logger import logger
from core.helpers.json_extractor import extract_json_array

class RagSystem:
    """
    Rag pipeline implemented as a workflow.
    For now, let's just make it a class before transitioning it into the workflow.
    """
    
    def __init__(
        self,
        router_llm: GoogleGenAI | None,
        chat_llm: GoogleGenAI | None,
        embed_model: HuggingFaceEmbedding | None,
        docs_path: str,
        collections: dict   
    ):
        self.router_llm = router_llm
        self.chat_llm = chat_llm
        self.embed_model = embed_model
        self.docs_path = docs_path
        self.collections = collections
        
        # We initialize it as an attribute since we won't need to recreate it many times
        self.router_retriever = self._get_router_retriever()
        
        # Cached chat engines per user   
        self.chat_engines_cached = {}
    
    
    def _ingest(self) -> list[RetrieverTool]:
        # Initialize the retriever_tools list to create a list of RetrieverTool objects that we will later
        # pass into the LLMMultiSelector for selecting an appropriate retriever
        retriever_tools = []

        # I manually wrote a dictionary for each course and its decription inside the previously loaded JSON file. 
        # 'collection_name' matches the name of the course folder inside the documents folder
        for collection_name, collection_description in self.collections.items():

            collection_path = self.docs_path + '/' + collection_name

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
        
        return retriever_tools
    
    
    def _get_router_retriever(self) -> RouterRetriever:
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
            retriever_tools = self._ingest()
        )
        return router
    
    
    async def _is_retrieval_relevant(self, user_query: str) -> str | bool:
        # Check the relevance of the nodes retrieved from router retriever.
        # We will return either False value if no retrieval is irrelevant or the retrieved
        # context otherwise
        try:
            retrieved_nodes = await self.router_retriever.aretrieve(user_query)
        except ValueError as e:
            logger.warning(f"{e}: knowledge base does not contain relevant info; no nodes were retrieved")
            return False
        
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
            return False
        
        # if not relevant_node_ids:
        #     logger.warning("Retrieved nodes do not containt a relevant info.")
        # else:
        #     pass

        context = "\n\n".join(
            [retrieved_nodes_dict[node_id] for node_id in relevant_node_ids]
        )
        return context
    
    
    def _get_or_create_chat_engine(self, user_id: str) -> SimpleChatEngine:
        # Here we either get cached chat_engine (SimpleChatEngine object) for a specific user 
        # or we create a new chat_engine for the user if he hadn't one before.
        if user_id in self.chat_engines_cached:
            logger.info("Using cached engine for the user {user_id}")
            return self.chat_engines_cached[user_id]
        
        logger.info("Creating a new chat engine for user {user_id}")
        
        memory = Memory.from_defaults(
            session_id = "Test_session_id", # change it later on a correct session_id
            token_limit = 2000,
            chat_history_token_ratio = 1.0
        )
        
        chat_engine = SimpleChatEngine.from_defaults(
            llm = self.chat_llm,
            memory = memory
        )
        self.chat_engines_cached[user_id] = chat_engine
        return chat_engine
    
    
    async def query(self, user_name: str, user_id: str, user_query: str) -> str:
        # Here we produce the final response to the user's query
        
        context = await self._is_retrieval_relevant(user_query)
        
        if not context:
            return f"{user_name}, I am sorry, but it seems that I don't have an answer to your question in my knowledge base, or it might be irrelevant."
        
        prompt = RagConstants.SYSTEM_PROMPT.format(
            user_name = user_name,
            question = user_query,
            context = context
        )
        
        chat_engine = self._get_or_create_chat_engine(user_id)
        
        response = await chat_engine.achat(prompt)
        return response.response
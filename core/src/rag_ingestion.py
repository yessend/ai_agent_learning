from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMMultiSelector

from core.config.config import Config
from core.config.constants import RagConstants
from core.config.llm_setup import LLMsetups

from core.helpers.logger import logger

class RagIngestion:
    # This is the RAG class to ingest the documents and form the knowledge base
    # It will be set up in the workflow as a dependency
    
    def __init__(self):
        self.router_llm = LLMsetups.ROUTER_LLM
        self.embed_model = LLMsetups.EMBED_MODEL
        self.docs_path = RagConstants.DOCS_PATH
        self.collections = RagConstants.COLLECTIONS


    def ingest(self) -> RetrieverTool | None:
        # Initialize the retriever_tools list to create a list of RetrieverTool objects that we will later
        # pass into the LLMMultiSelector for selecting an appropriate retriever
        
        if not self.docs_path or not self.collections:
            logger.warning("Ingestion method cannot be performed, missing arguments in the Start Event.")
            return None
        
        retriever_tools = []

        # I manually wrote a dictionary for each course and its decription inside the previously loaded JSON file. 
        # 'collection_name' matches the name of the course folder inside the documents folder
        for collection_name, collection_description in self.collections.items():

            collection_path = self.docs_path / collection_name

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
            retriever_tools = retriever_tools
        )
        return router
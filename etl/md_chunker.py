from llama_index.core.node_parser import MarkdownNodeParser, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader

from llama_index.core.schema import Document, BaseNode

embed_model = HuggingFaceEmbedding(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

def hybrid_hierarchical_chunking(dir_path):
    """
    Takes raw Markdown documents and turns them into Raj-style "Smart Chunks".
    """
    
    # STEP 1: STRUCTURE (Raj's Level 1 & 2)
    # Split by Markdown Headers (#, ##, ###)
    # This guarantees that "Installation" never leaks into "Troubleshooting".
    reader = SimpleDirectoryReader(dir_path)
    documents = reader.load_data()
    markdown_parser = MarkdownNodeParser()
    base_nodes = markdown_parser.get_nodes_from_documents(documents)
    
    # STEP 2: MEANING (Raj's Level 3)
    # Now look at each section. If it's huge, split it by MEANING, not character count.
    semantic_splitter = SemanticSplitterNodeParser.from_defaults(
        buffer_size=1, 
        breakpoint_percentile_threshold=95,  # The "Sensitivity" of the cut
        embed_model=embed_model
    )
    
    final_nodes = []
    
    for node in base_nodes:
        # If the section is small (e.g. < 500 chars), just keep it.
        if len(node.text) < 500:
            final_nodes.append(node)
            continue
            
        # If the section is huge, run Semantic Splitter on just this node's text
        semantic_nodes = semantic_splitter.get_nodes_from_documents([
            Document(text=node.get_content(), metadata=node.metadata)
        ])
        
        # CRITICAL: Preserve the Metadata!
        # The semantic splitter might lose the "Header Path" from Step 1.
        # We must copy it back to the children.
        for child in semantic_nodes:
            # Copy parent metadata (e.g. {'header_path': 'Setup > Docker'})
            child.metadata = node.metadata.copy()
            
            # Add "Raj's Level 4" magic: Inject context into the text
            # The LLM sees: "Context: Setup > Docker. Content: Set sparse to true..."
            header_context = child.metadata.get("header_path", "")
            child.text = f"Context: {header_context}\nContent: {child.text}"
            
            final_nodes.append(child)
            
    return final_nodes


def markdown_parser(doc_path):
    return
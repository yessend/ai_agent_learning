import json
from etl.core.config import ETLConfig

from llama_index.core.schema import TextNode, IndexNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient


def main():
    chunk_files = list(ETLConfig.CHUNKED_DIR.glob("*.json"))
    parents_chunks_file = chunk_files[-1]
    children_chunks_file = chunk_files[-2]

    with open(parents_chunks_file, "r", encoding="utf-8") as f:
        parent_nodes_list = json.load(f)
        
    with open(children_chunks_file, "r", encoding="utf-8") as f:
        child_nodes_list = json.load(f)

    parent_mapping = {}
    parent_nodes = []
    child_nodes = []

    for section_node in parent_nodes_list:
        node = TextNode(
        id_ = section_node["id_"],
        text = section_node["text"],
        metadata = section_node["metadata"]
        )
        parent_mapping[section_node["id_"]] = node
        parent_nodes.append(node)

    for child_node in child_nodes_list:
        node = IndexNode(
            id_ = child_node["id_"],
            text = child_node["text"],
            metadata = child_node["metadata"],
            excluded_embed_metadata_keys = child_node["excluded_embed_metadata_keys"],
            index_id = child_node["index_id"]
        )
        child_nodes.append(node)

    client = QdrantClient(url = ETLConfig.QDRANT_URL)

    embed_model = HuggingFaceEmbedding(
        model_name=ETLConfig.EMBEDDING_MODEL
    )

    docstore = SimpleDocumentStore().add_documents(parent_nodes)

    vector_store = QdrantVectorStore(
        client=client, 
        collection_name=ETLConfig.COLLECTION_NAME,
        enable_hybrid=True,
        dense_vector_name="dense_embed_vector",
        sparse_vector_name="sparse_keyword_vector",
        fastembed_sparse_model="Qdrant/bm25"
    )

    storage_context = StorageContext.from_defaults(
        docstore=docstore,
        vector_store=vector_store
    )

    nodes_to_embed = child_nodes[:10]

    vector_index = VectorStoreIndex(
        nodes=nodes_to_embed,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

if __name__ == "__main__":
    main()
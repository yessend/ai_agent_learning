import json
from pathlib import Path
from etl.core.config import ETLConfig

from llama_index.core.schema import TextNode, IndexNode
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from qdrant_client import QdrantClient, AsyncQdrantClient, async_qdrant_client
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    Distance,
    Modifier,
    PointStruct,
    Document
)
from sentence_transformers import SentenceTransformer


class QdrantVectorDB:

    def __init__(
            self,
            qdrant_client: QdrantClient,
            async_qdrant_client: AsyncQdrantClient,
            embed_model: SentenceTransformer,
            embed_dim: int,
            collection_name: str
        ) -> None:
        self.qdrant_client = qdrant_client
        self.async_qdrant_client = async_qdrant_client
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.collection_name = collection_name
    

    async def _get_collection(self) -> bool:
        if not await self.async_qdrant_client.collection_exists(self.collection_name):
            await self.async_qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        distance=Distance.COSINE,
                        size=self.embed_dim
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        modifier=Modifier.IDF
                    )
                }
            )
            
    
    async def ingest_points(self, file_path: Path):
        with open(file_path, "r", encoding="utf-8") as file:
            nodes = json.load(file)
        await self.async_qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=node["id_"],
                    vector={
                        "dense": self.embed_model.encode_document(node["text"]).tolist(),
                        "sparse": Document(
                            text=node["text"],
                            model="Qdrant/bm25"
                        )
                    },
                    payload={
                        "text": node["text"],
                        "metadata": node["metadata"]
                    }
                )
                for node in nodes
            ]
        )


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
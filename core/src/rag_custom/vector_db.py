from qdrant_client import AsyncQdrantClient


class AsyncQdrantVectorDB:

    def __init__(self, aclient: AsyncQdrantClient, embed_dim: int) -> None:
        self.aclient = aclient,
        self.embed_dim = embed_dim
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

# These are the custom classes to perform RAG and synthesis of an answer:
class RagGenerateEvent(Event):
    retrieved_nodes: list[NodeWithScore] | None
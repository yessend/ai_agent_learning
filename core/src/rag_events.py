from llama_index.core.workflow import Event

# These are the custom classes to perform RAG and synthesis of an answer:
class RetrievalRelevantEvent(Event):
    context: str | bool
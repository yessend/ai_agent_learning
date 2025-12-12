import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CHAT_LLM = "gemini-2.5-flash-lite"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 3000
    
    EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
    
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
    
    SIMILARITY_TOP_K = 5
    ROUTER_RETRIEVER_MAX_OUTPUTS = 3
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    CHAT_LLM = "gemini-2.5-flash-lite"
    CHAT_LLM_TEMPERATURE = 0.1
    CHAT_LLM_MAX_TOKENS = 3000

    ROUTER_LLM = "gemini-2.5-flash-lite"
    ROUTER_LLM_TEMPERATURE = 0.1
    ROUTER_LLM_MAX_TOKENS = 2500
    
    EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
    
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
    
    SIMILARITY_TOP_K = 5
    ROUTER_RETRIEVER_MAX_OUTPUTS = 3
    
    CHAT_MEMORY_TOKEN_LIMIT = 2000
    GROUNDING_MAX_OUTPUT_TOKENS = 3000
    GROUNDING_LAST_N_MESSAGES = -6
    CHAT_HISTORY_TOKEN_RATIO = 1.0
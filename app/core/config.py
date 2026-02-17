import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    CHAT_LLM = "gemini-2.5-flash-lite"
    CHAT_LLM_TEMPERATURE = 0.1
    CHAT_LLM_MAX_TOKENS = 3000
    CHAT_THINKING_BUDGET = 2048
    
    EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
    
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
    QDRANT_COLLECTION_NAME = "some_collection"
    
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT"))
    REDIS_URL = os.getenv("REDIS_URL")
    REDIS_MAX_CONNECTIONS = 100
    REDIS_TIMEOUT = 5 # in seconds
    REDIS_TTL = 3600 # in seconds (make it bigger for the production)

    SIMILARITY_TOP_K = 20
    
    CHAT_HISTORY_TOKEN_LIMIT = 2000
    CHAT_HISTORY_FETCH_LIMIT = 40
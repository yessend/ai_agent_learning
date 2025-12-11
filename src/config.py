import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CHAT_LLM = "gemini-2.5-flash-lite"
    EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
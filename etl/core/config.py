from pathlib import Path
import os

class ETLConfig:
    
    DATA_DIR = Path("./data")
    RAW_DIR = DATA_DIR / "pdfs"
    PARSED_DIR = DATA_DIR / "parsed"
    CLEANED_DIR = DATA_DIR / "cleaned"
    CHUNKED_DIR = DATA_DIR / "chunked"
    
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
    COLLECTION_NAME = "talapkerai"
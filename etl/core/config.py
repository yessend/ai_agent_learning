from pathlib import Path
import os

class ETLConfig:
    
    DATA_DIR = Path("./data")
    RAW_DIR = DATA_DIR / "01_raw"
    PARSED_DIR = DATA_DIR / "02_parsed"
    CLEANED_DIR = DATA_DIR / "03_cleaned"
    CHUNKED_DIR = DATA_DIR / "04_chunked"
    
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
    COLLECTION_NAME = "talapkerai"
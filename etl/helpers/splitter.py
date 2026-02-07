import json
from pathlib import Path
from typing import List
from transformers import AutoTokenizer
from etl.core.config import ETLConfig
from helpers.logger import logger
import uuid

class TextSplitter:
    def __init__(
            self,
            model_name: str,
            chunk_size: int = 512,
            chunk_overlap: int = 50
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]

    def _get_token_length(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _hard_split(self, text: str) -> List[str]:
        """Fallback for extremely long strings without separators."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return [
            self.tokenizer.decode(tokens[i : i + self.chunk_size])
            for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap)
        ]
        
    def split_text(self, text: str) -> List[str]:
        """
        Recursively splits text based on separators while respecting token limits.
        """
        if self._get_token_length(text) <= self.chunk_size:
            return [text]

        separator = " "
        for s in self.separators:
            if s in text:
                separator = s
                break

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            test_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if self._get_token_length(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if self._get_token_length(split) > self.chunk_size:
                    chunks.extend(self._hard_split(split))
                    current_chunk = ""
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks


    def run_chunking(self, input_path: Path, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = list(input_path.glob("*.md"))
        
        for file_path in files:
            content = file_path.read_text(encoding="utf-8")
            
            # Split logic
            text_chunks = self.split_text(content)
            
            chunks_data = []
            for i, chunk in enumerate(text_chunks):
                chunks_data.append({
                    "node_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk)),
                    "text": chunk,
                    "metadata": {
                        "file_name": file_path.name,
                        "chunk_index": i
                    }
                })
            
            output_file = output_path / f"{file_path.stem}_chunks.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"✂️ Vanilla Chunking: {file_path.name} -> {len(chunks_data)} chunks")
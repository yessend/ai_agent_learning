import nest_asyncio
import asyncio

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.google_genai import GoogleGenAI

import hashlib
import json

from core.config.config import Config
from helpers.logger import logger
from etl.core.constants import ETLConstants
from etl.core.config import ETLConfig
from etl.helpers.helpers import (
    KeywordList,
    get_file_summary_lang,
    get_section_summary,
    generate_deterministic_uuid,
    generate_dense_context
)

nest_asyncio.apply()


async def main():
    # define some constants
    md_files = ETLConfig.CLEANED_DIR
    files = list(md_files.glob("*.md"))
    junk_metadata = ["doc_summary", "section_summary", "file_path", "directory", "file_name", "file_type", "custom_id"]

    parent_nodes = []
    child_nodes = []
    log_docs = {
        "success_files": 0,
        "failed_files": 0,
        "success_splits": 0,
        "failed_splits": 0
    }
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4"),
        ("#####", "Header_5"),
        ("######", "Header_6"),
    ]

    llm = GoogleGenAI(
            model=Config.ROUTER_LLM,
            api_key=Config.GOOGLE_API_KEY,
            temperature=0.1
        )

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=KeywordList,
        prompt_template_str=ETLConstants.KEYWORD_EXTRACTOR_PROMPT,
        llm=llm
    )

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True
    )

    lc_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    logger.info(f"Read files from the {str(md_files.absolute())} directory...")

    for file in files:
        try:
            md_content = file.read_text(encoding="utf-8")
            md_header_splits = splitter.split_text(md_content)
            
            logger.info(f"Generating summary for the {file.name} document...")
            file_summ = await get_file_summary_lang(llm, md_content)
            
            logger.info(f"Generating chunks for {file.name} document with rich metadata...")
            for split in md_header_splits:
                try:
                    text = split.page_content
                    section_summary = await get_section_summary(llm, text, file_summ["language"])
                    headers = split.metadata
                    id_ = hashlib.sha256(text.encode()).hexdigest()
                    metadata = {
                        "doc_summary": file_summ["summary"],
                        "section_summary": section_summary,
                        "file_name": file.name,
                        "file_type": file.suffix,
                        "directory": file.parent.name,
                        "file_path": str(file.absolute()),
                        "language": file_summ["language"],
                        **headers
                    }
                    parent_nodes.append({
                        "id_": id_,
                        "text": text,
                        "metadata": metadata
                    })
                    
                    child_texts = lc_splitter.split_text(text)
                    for i, child_text in enumerate(child_texts):
                        custom_id = f"{id_}_child_{i}"
                        qdrant_id = generate_deterministic_uuid(custom_id)
                        output = program(
                            node_text = child_text,
                            language = "kazakh" if file_summ["language"] == "kk" else "russian"
                        )
                        dense_context = await generate_dense_context(llm, text, child_text, file_summ["language"])
                        child_nodes.append({
                            "id_": qdrant_id,
                            "text": child_text,
                            "metadata": {
                                **metadata,
                                "dense_context": dense_context,
                                "excerpt_keywords": output.keywords,
                                "custom_id": custom_id
                            },
                            "excluded_embed_metadata_keys": junk_metadata,
                            "index_id": id_
                        })
                    
                    log_docs["success_splits"] += 1
                except Exception as e:
                    logger.error(f"Error chunking the {file.name} document, certain split processing failed: {e}...")
                    logger.info("Skipping to the next split of the current document...")
                    log_docs["failed_splits"] += 1
                    continue
            log_docs["success_files"] += 1
                
        except Exception as e:
            logger.error(f"Error processing the document {file.name}: {e}...")
            logger.info("Skipping to the next document...")
            log_docs["failed_files"] += 1
            continue

    logger.info(f"Successfully processed: {log_docs["success_files"]} files...")
    logger.warning(f"Failed to process {log_docs["failed_files"]} files...")

    output_file_parents = ETLConfig.CHUNKED_DIR / "parent_chunks_ver2.json"
    logger.info(f"Saving parent nodes to the file {output_file_parents}...")
    with open(output_file_parents.absolute().resolve(), "w", encoding="utf-8") as f:
        json.dump(parent_nodes, f, ensure_ascii=False, indent=2)
        
    output_file_children = ETLConfig.CHUNKED_DIR / "children_chunks_ver2.json"
    logger.info(f"Saving parent nodes to the file {output_file_children}...")
    with open(output_file_children.absolute().resolve(), "w", encoding="utf-8") as f:
        json.dump(child_nodes, f, ensure_ascii=False, indent=2)
    
if __name__ == "__main__":
    asyncio.run(main())
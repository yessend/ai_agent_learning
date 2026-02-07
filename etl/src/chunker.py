import nest_asyncio
import asyncio

from etl.helpers.splitter import TextSplitter
from google import genai

import uuid
import json
from tqdm import tqdm
from time import sleep

from core.config.config import Config
from etl.helpers.logger import logger
from etl.core.config import ETLConfig
from etl.helpers.helpers import (
    get_file_summary_lang,
    get_chunk_keywords,
    get_section_summary,
    generate_dense_context
)

nest_asyncio.apply()


async def main():
    # define some constants
    md_files = ETLConfig.PARSED_DIR
    files = list(md_files.glob("*.md"))
    
    nodes = []
    log_docs = {
        "success_files": 0,
        "failed_files": 0,
        "success_chunks": 0,
        "failed_chunks": 0
    }

    llm_client = genai.Client(api_key=Config.GOOGLE_API_KEY)

    splitter = TextSplitter(
        model_name=ETLConfig.EMBEDDING_MODEL,
        chunk_size=512, 
        chunk_overlap=75
    )

    logger.info(f"Read files from the {str(md_files.absolute())} directory...")

    max_retries = 5
    delay_sec = 5

    for file in tqdm(files, desc="Processing .md files", position=0, unit="file"):

        success_file = False
        attempts_file = 0

        while attempts_file < max_retries and not success_file:
            try:
                md_content = file.read_text(encoding="utf-8")
                chunks = splitter.split_text(md_content)
                
                logger.info(f"Generating summary for the {file.name} document...")
                file_summ = await get_file_summary_lang(llm_client, md_content)
                
                logger.info(f"Generating chunks for {file.name} document with rich metadata...")

                for i, chunk in enumerate(tqdm(chunks, desc=f"Chunking {file.name} file", position=1, unit="chunk")):

                    success_chunk = False
                    attempts_chunk = 0

                    if len(chunk.strip()) < 25:
                        logger.info(f"Skipping the chunk {i} of file {file.name} because it's too small...")
                        continue

                    while attempts_chunk < max_retries and not success_chunk:
                        try:
                            text = chunk
                            section_summary = await get_section_summary(llm_client, text, file_summ["language"])
                            keywords = await get_chunk_keywords(
                                llm_client=llm_client,
                                node_text = chunk,
                                language = file_summ["language"]
                            )
                            dense_context = await generate_dense_context(llm_client, text, chunk, file_summ["language"])
                            # headers = split.metadata
                            id_ = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
                            metadata = {
                                "doc_summary": file_summ["summary"],
                                "section_summary": section_summary,
                                "file_name": file.name,
                                "file_type": file.suffix,
                                "directory": file.parent.name,
                                "file_path": str(file),
                                "language": file_summ["language"],
                                "chunk_id": i,
                                "dense_context": dense_context,
                                "excerpt_keywords": keywords["keywords"],
                                # **headers
                            }
                            nodes.append({
                                "id_": id_,
                                "text": f"{dense_context}\n\n{chunk}",
                                "metadata": metadata,
                                "excluded_embed_metadata_keys": list(metadata.keys()),
                            })

                            log_docs["success_chunks"] += 1
                            success_chunk = True

                        except Exception as e:
                            logger.error(f"Error chunking the {file.name} document, certain split processing failed: {e}...")
                            logger.info("Skipping to the next split of the current document...")
                            attempts_chunk += 1
                            if attempts_chunk < max_retries:
                                logger.info(f"Retry to process certain split in {file.name} document in {delay_sec * attempts_chunk} seconds...")
                                sleep(delay_sec * attempts_chunk)
                            else:
                                logger.info(f"Skipping to the next split of the {file.name} document after {max_retries} failed attempts.")
                                log_docs["failed_chunks"] += 1
                                continue

                log_docs["success_files"] += 1
                success_file = True
                    
            except Exception as e:
                logger.error(f"Error processing the document {file.name}: {e}...")
                logger.info("Skipping to the next document...")
                attempts_file += 1
                if attempts_file < max_retries:
                    logger.info(f"Retry to process certain split in {file.name} document in {delay_sec * attempts_file} seconds...")
                    sleep(delay_sec * attempts_file)
                else:
                    logger.info(f"Skipping to the next split of the {file.name} document after {max_retries} failed attempts.")
                    log_docs["failed_files"] += 1
                    continue

    logger.info(f"Successfully processed: {log_docs["success_files"]} files...")
    logger.info(f"Failed to process {log_docs["failed_files"]} files...")

    ETLConfig.CHUNKED_DIR.mkdir(parents=True, exist_ok=True)
    output_chunks = ETLConfig.CHUNKED_DIR / "chunks_from_pdfs.json"
    logger.info(f"Saving parent nodes to the file {output_chunks}...")
    with open(output_chunks.absolute().resolve(), "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)

    
if __name__ == "__main__":
    asyncio.run(main())
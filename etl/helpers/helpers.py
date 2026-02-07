import json
import hashlib
from uuid import UUID
from etl.core.constants import ETLConstants
from core.config.config import Config
from google.genai import Client, types

async def get_file_summary_lang(llm_client: Client, full_text):
    response = await llm_client.aio.models.generate_content(
        model=Config.CHAT_LLM,
        contents=ETLConstants.DOCUMENT_SUMMARY_EXTRACTOR_PROPMT.format(full_text=full_text),
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    return json.loads(response.text.strip().replace("```json", "").replace("```", ""))

async def get_chunk_keywords(llm_client: Client, node_text, language):
    response = await llm_client.aio.models.generate_content(
        model=Config.CHAT_LLM,
        contents=ETLConstants.KEYWORD_EXTRACTOR_PROMPT.format(
            node_text=node_text,
            language="kazakh" if language == "kk" else "russian" if language == "ru" else "english"
        ),
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
    )
    return json.loads(response.text.strip().replace("```json", "").replace("```", ""))

async def get_section_summary(llm_client: Client, full_text, language):
    response = await llm_client.aio.models.generate_content(
        model=Config.CHAT_LLM,
        contents=ETLConstants.SECTION_SUMMARY_EXTRACTOR_PROPMT.format(
            full_text=full_text,
            language="kazakh" if language == "kk" else "russian" if language == "ru" else "english"
        ),
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    return response.text.strip()

def generate_deterministic_uuid(content_string: str) -> str:
    hash_obj = hashlib.md5(content_string.encode("utf-8"))
    return str(UUID(hash_obj.hexdigest()))

async def generate_dense_context(llm_client: Client, section, child_text, language) -> str:
    response = await llm_client.aio.models.generate_content(
        model=Config.CHAT_LLM,
        contents=ETLConstants.SITUATIONAL_CONTEXT_PROMPT.format(
            section=section,
            child_text=child_text,
            language="kazakh" if language == "kk" else "russian" if language == "ru" else "english"
        ),
        config=types.GenerateContentConfig(
            temperature=0.1
        )
    )
    return response.text.strip()
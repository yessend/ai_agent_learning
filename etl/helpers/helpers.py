from pydantic import BaseModel, Field
from typing import List
import json
import hashlib
from uuid import UUID
from etl.core.constants import ETLConstants

class KeywordList(BaseModel):
    """Data model for a list of keywords."""
    keywords: List[str] = Field(description="A list of exactly 5 search-optimized keywords.")

async def get_file_summary_lang(llm, full_text):
    response = await llm.acomplete(ETLConstants.DOCUMENT_SUMMARY_EXTRACTOR_PROPMT.format(full_text=full_text))
    return json.loads(response.text.strip().replace("```json", "").replace("```", ""))

async def get_section_summary(llm, full_text, language):
    response = await llm.acomplete(ETLConstants.SECTION_SUMMARY_EXTRACTOR_PROPMT.format(full_text=full_text, language="kazakh" if language == "kk" else "russian"))
    return response.text.strip()

def generate_deterministic_uuid(content_string: str) -> str:
    hash_obj = hashlib.md5(content_string.encode("utf-8"))
    return str(UUID(hash_obj.hexdigest()))

async def generate_dense_context(llm, section, child_text, language) -> str:
    response = await llm.acomplete(ETLConstants.SITUATIONAL_CONTEXT_PROMPT.format(section=section, child_text=child_text, language="kazakh" if language == "kk" else "russian"))
    return response.text.strip()
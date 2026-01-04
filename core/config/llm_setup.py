from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from google.genai import types

from core.config.config import Config

class LLMsetups:
    ROUTER_LLM = GoogleGenAI(
        model = Config.ROUTER_LLM,
        api_key = Config.GOOGLE_API_KEY,
        generation_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(thinking_budget = 0),
            temperature = Config.ROUTER_LLM_TEMPERATURE,
        ),
        max_tokens = Config.ROUTER_LLM_MAX_TOKENS
    )

    CHAT_LLM = GoogleGenAI(
        model = Config.CHAT_LLM,
        api_key = Config.GOOGLE_API_KEY,
        generation_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(thinking_budget = 0),
            temperature = Config.CHAT_LLM_TEMPERATURE,
        ),
        max_tokens = Config.CHAT_LLM_MAX_TOKENS
    )

    EMBED_MODEL = HuggingFaceEmbedding(
        model_name = Config.EMBEDDING_MODEL
    )
import json

from app.core.config import Config

from google.genai import Client, types
import redis.asyncio as async_redis


class GeminiChatEngine():
    
    def __init__(
        self,
        llm_client: Client,
        system_prompt: str,
        redis_async_client: async_redis.Redis | None, 
        redis_store_key: str,
        history_fetch_limit: int,
        history_token_limit: int
    ):
        self.llm_client = llm_client
        self.gen_config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            system_instruction=system_prompt,
            temperature=Config.CHAT_LLM_TEMPERATURE,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_budget=Config.CHAT_THINKING_BUDGET
            )
        )

        self.redis_async_client = redis_async_client
        self.redis_store_key = redis_store_key
        
        self.history_fetch_limit = history_fetch_limit              # How many messages to retrieve from the redis store at once
        self.history_token_limit = history_token_limit
    
    
    async def _contents_to_dict(self, contents: list[types.Content | types.GenerateContentResponse]) -> list[str]:
        dict_temp_list = []
        for content in contents:
            dict_temp = {}
            if isinstance(content, types.Content):
                dict_temp["role"] = content.role
                dict_temp["content"] = content.parts[0].text
            else:
                dict_temp["role"] = content.candidates[0].content.role
                dict_temp["content"] = content.candidates[0].content.parts[1].text
                thought_process = getattr(content.candidates[0].content.parts[0], 'thought', "No thinking data available")
                dict_temp["thought_process"] = content.candidates[0].content.parts[0].text if thought_process else thought_process
                usage_tokens = {
                    'prompt_tokens': content.usage_metadata.prompt_token_count,
                    'completion_tokens': content.usage_metadata.candidates_token_count,
                    'thoughts_token_count': content.usage_metadata.thoughts_token_count,
                    'total_tokens': content.usage_metadata.total_token_count
                }
                dict_temp["usage_metadata"] = usage_tokens
            dict_temp_list.append(json.dumps(dict_temp))
        return dict_temp_list
    
    
    async def _dict_to_contents(self, redis_dicts: list[dict]) -> list[types.Content]:
        return [types.Content(role=dict_content.get("role"), parts=[types.Part(text=dict_content.get("content"))]) for dict_content in redis_dicts]

    
    async def _get_history_safe(self) -> list[types.Content]:
    
        history = await self.redis_async_client.lrange(self.redis_store_key, -self.history_fetch_limit, -1)
        if len(history) == 0:
            return []

        history_list = [json.loads(m.decode("utf-8")) for m in history]

        while history_list and history_list[0].get("role") != "user":
            history_list.pop(0)
                
        while history_list and history_list[-1].get("role") != "model":
            history_list.pop()
        
        history_list_inv = history_list[::-1]
        
        total_tokens = 0
        number_of_interactions = 0
        for message in history_list_inv:
            if "usage_metadata" in message:
                total_tokens += message["usage_metadata"].get("prompt_tokens") + message["usage_metadata"].get("completion_tokens")
            if total_tokens >= self.history_token_limit:
                break
            else:
                number_of_interactions += 2
        
        return await self._dict_to_contents(history_list[-number_of_interactions:])
    
    
    async def achat(
        self, 
        message: str, 
        user_name: str,
        answer_lang: str,
        context: str | None,
    ) -> str:
        
        query_full = f"""
            User's name: {user_name}
            Answer in {answer_lang} language
            Question: {message}

            {f"Use the context information below to answer user's question.\n<context>\n{context}\n<context>\n" if context else ""}
        """
        
        query_push = types.Content(role="user", parts=[types.Part(text=message)])
        query_wrapped = types.Content(role="user", parts=[types.Part(text=query_full)])

        chat_history = await self._get_history_safe()
        
        contents = chat_history + [query_wrapped]
        
        response = await self.llm_client.aio.models.generate_content(
            model=Config.CHAT_LLM,
            contents=contents,
            config=self.gen_config
        )
        
        ai_message = response
        
        messages_to_push = await self._contents_to_dict([query_push, response])
        await self.redis_async_client.rpush(self.redis_store_key, *messages_to_push)
        await self.redis_async_client.expire(self.redis_store_key, Config.REDIS_TTL)

        return ai_message
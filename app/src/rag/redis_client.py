import redis.asyncio as async_redis
from app.core.config import Config

class RedisClient:
    def redis_pool_init(self) -> async_redis.Redis:

        async_pool = async_redis.BlockingConnectionPool.from_url(
            Config.REDIS_URL, 
            max_connections = Config.REDIS_MAX_CONNECTIONS,
            timeout = Config.REDIS_TIMEOUT, 
            decode_responses = False
        )

        redis_async_client = async_redis.Redis(connection_pool = async_pool)

        return redis_async_client
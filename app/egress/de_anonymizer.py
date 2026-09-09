import json
from pyexpat import errors
import re
import redis.asyncio as redis
from typing import Optional


class DeAnonymizerService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def restore(self, masked_text: str, session_key: str) -> str:
        """
        Reconstructs the original text by replacing surrogate keys
        with actual PII values from Redis.
        """

        redis_key = f"rev_map_session:{session_key}"
        raw_map = await self.redis.get(redis_key)

        if not raw_map:
            return masked_text
        #1. using the session key fetch the map
        #2. replace in masked_text return new string text
        #3. return the output

        # Decode JSON stored during Ingress
        if isinstance(raw_map, bytes):
            reverse_map = json.loads(raw_map.decode("utf-8",errors='strict'))
        else:
            reverse_map = json.loads(raw_map)

        if not reverse_map:
            return masked_text

        pattern = "|".join(re.escape(k) for k in reverse_map.keys())

        surrogate_pattern = re.compile(pattern)
        restored_text = surrogate_pattern.sub(lambda match: reverse_map[match.group(0)],masked_text)
        return restored_text
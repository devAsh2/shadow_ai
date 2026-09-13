from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
from app.ingress.sanitizer import SanitizerService
from app.cache.createEmbedding import EmbeddingService
from app.cache.semantic_cache import SemanticCacheService
from app.cache.pg_cache import PostgresCacheService
from app.agent.graph import agent_graph
from app.egress.de_anonymizer import DeAnonymizerService
import logging

logger = logging.getLogger(__name__)

class ProxyService:
    def __init__(
        self,
        sanitizer: SanitizerService,
        embedding_service: EmbeddingService,
        redis_cache: SemanticCacheService,
        pg_cache: PostgresCacheService,
        de_anonymizer: DeAnonymizerService,
    ):
        self.sanitizer = sanitizer
        self.embedding_service = embedding_service
        self.redis_cache = redis_cache
        self.pg_cache = pg_cache
        self.de_anonymizer = de_anonymizer

    async def handle_prompt(self, raw_prompt: str) -> dict:
        # 1. Ingress Sanitizer: Mask PII and store reverse mapping in Redis
        sanitized_data = await self.sanitizer.sanitize_and_store(raw_prompt)
        masked_prompt = sanitized_data["masked_prompt"]
        session_key = sanitized_data["session_key"]

        # 2. Embedding Service: Generate vector representation (Single Model Execution)
        vector_bytes, vector_list = await self.embedding_service.get_embeddings(masked_prompt)

        # 3. Tier 1: Check Redis RAM Cache (Sub-20ms lookup)
        cached_response = await self.redis_cache.check_cache(vector_bytes)
        if cached_response:
            logger.info("Proxy response source=redis_cache session=%s", session_key)
            return {
                "source": "redis_cache",
                "response": await self.de_anonymizer.restore(cached_response, session_key),
                "session_key": session_key,
            }

        # 4. Tier 2: Check Postgres pgvector Cache (Long-term persistent store)
        cached_response = await self.pg_cache.check_cache(vector_list)
        if cached_response:
            logger.info("Proxy response source=postgres_cache session=%s", session_key)
            # Cache Warming: Write back to Redis with TTL so subsequent calls hit Tier 1
            await self.redis_cache.set_cache(
                masked_prompt=masked_prompt,
                masked_response=cached_response,
                query_vector=vector_bytes,
                ttl=86400,
            )
            return {
                "source": "postgres_cache",
                "response": await self.de_anonymizer.restore(cached_response, session_key),
                "session_key": session_key,
            }

        # 5. Full Cache Miss -> Invoke LangGraph Agent Engine
        initial_state = {
            "masked_prompt": masked_prompt,
            "session_key": session_key,
            "messages": [],
            "final_response": "",
        }
        
        agent_result = await agent_graph.ainvoke(initial_state)
        masked_response = agent_result["final_response"]
        logger.info("Proxy response source=agent_engine session=%s", session_key)

        # 6. Asynchronous Write-Back (Store in Redis & Postgres simultaneously)
        await asyncio.gather(
            self.redis_cache.set_cache(
                masked_prompt=masked_prompt,
                masked_response=masked_response,
                query_vector=vector_bytes,
                ttl=86400,
            ),
            self.pg_cache.set_cache(
                masked_prompt=masked_prompt,
                masked_response=masked_response,
                embedding=vector_list,
            ),
        )

        final_text = await self.de_anonymizer.restore(masked_response,session_key)

        return {
            "source": "agent_engine",
            "response": final_text,
            "session_key": session_key,
        }
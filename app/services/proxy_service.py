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

    async def handle_prompt(
        self,
        raw_prompt: str,
        session_key: str,
        langfuse_callback=None,
        trace=None,
    ) -> dict:
        trace_session_key = session_key

        # 1. Ingress Sanitizer: Mask PII and store reverse mapping in Redis
        sanitizer_span = trace.span(
            name="ingress_sanitizer",
            metadata={"session_key": trace_session_key},
        ) if trace else None
        try:
            sanitized_data = await self.sanitizer.sanitize_and_store(raw_prompt)
        except Exception as exc:
            if sanitizer_span:
                sanitizer_span.update(status_message=str(exc), level="ERROR")
            raise

        masked_prompt = sanitized_data["masked_prompt"]
        session_key = sanitized_data["session_key"]
        if sanitizer_span:
            sanitizer_span.update(
                output={"masked_prompt": masked_prompt},
                metadata={
                    "session_key": session_key,
                    "has_pii": sanitized_data["has_pii"],
                },
            )
        if trace:
            trace.update(metadata={"session_key": session_key})

        # 2. Embedding Service: Generate vector representation (Single Model Execution)
        embedding_span = trace.span(
            name="embedding_generation",
            input={"masked_prompt": masked_prompt},
            metadata={"session_key": session_key},
        ) if trace else None
        try:
            vector_bytes, vector_list = await self.embedding_service.get_embeddings(masked_prompt)
        except Exception as exc:
            if embedding_span:
                embedding_span.update(status_message=str(exc), level="ERROR")
            raise
        if embedding_span:
            embedding_span.update(
                output={"embedding_dimensions": len(vector_list)},
                metadata={"session_key": session_key},
            )

        # 3. Tier 1: Check Redis RAM Cache (Sub-20ms lookup)
        redis_span = trace.span(
            name="redis_cache_lookup",
            input={"embedding_dimensions": len(vector_list)},
            metadata={"session_key": session_key},
        ) if trace else None
        try:
            cached_response = await self.redis_cache.check_cache(vector_bytes)
        except Exception as exc:
            if redis_span:
                redis_span.update(status_message=str(exc), level="ERROR")
            raise
        if redis_span:
            redis_span.update(
                output={"cache_hit": bool(cached_response)},
                metadata={"session_key": session_key},
            )
        if cached_response:
            logger.info("Proxy response source=redis_cache session=%s", session_key)
            if trace:
                trace.update(
                    output={"source": "redis_cache"},
                    metadata={"session_key": session_key, "cache_hit": True},
                )
            return {
                "source": "redis_cache",
                "response": await self.de_anonymizer.restore(cached_response, session_key),
                "session_key": session_key,
            }

        # 4. Tier 2: Check Postgres pgvector Cache (Long-term persistent store)
        postgres_span = trace.span(
            name="postgres_cache_lookup",
            input={"embedding_dimensions": len(vector_list)},
            metadata={"session_key": session_key},
        ) if trace else None
        try:
            cached_response = await self.pg_cache.check_cache(vector_list)
        except Exception as exc:
            if postgres_span:
                postgres_span.update(status_message=str(exc), level="ERROR")
            raise
        if postgres_span:
            postgres_span.update(
                output={"cache_hit": bool(cached_response)},
                metadata={"session_key": session_key},
            )
        if cached_response:
            logger.info("Proxy response source=postgres_cache session=%s", session_key)
            # Cache Warming: Write back to Redis with TTL so subsequent calls hit Tier 1
            await self.redis_cache.set_cache(
                masked_prompt=masked_prompt,
                masked_response=cached_response,
                query_vector=vector_bytes,
                ttl=86400,
            )
            if trace:
                trace.update(
                    output={"source": "postgres_cache"},
                    metadata={"session_key": session_key, "cache_hit": True},
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

        agent_span = trace.span(
            name="agent_engine",
            input={"masked_prompt": masked_prompt},
            metadata={"session_key": session_key},
        ) if trace else None
        try:
            agent_config = {
                "metadata": {"session_key": session_key},
                "configurable": {"session_key": session_key},
            }
            if langfuse_callback:
                agent_config["callbacks"] = [langfuse_callback]
            agent_result = await agent_graph.ainvoke(initial_state, config=agent_config)
        except Exception as exc:
            if agent_span:
                agent_span.update(status_message=str(exc), level="ERROR")
            raise
        masked_response = agent_result["final_response"]
        if agent_span:
            agent_span.update(
                output={"masked_response": masked_response},
                metadata={"session_key": session_key},
            )
        logger.info("Proxy response source=agent_engine session=%s", session_key)

        # 6. Asynchronous Write-Back (Store in Redis & Postgres simultaneously)
        writeback_span = trace.span(
            name="cache_writeback",
            input={"masked_prompt": masked_prompt},
            metadata={"session_key": session_key},
        ) if trace else None
        try:
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
        except Exception as exc:
            if writeback_span:
                writeback_span.update(status_message=str(exc), level="ERROR")
            raise
        if writeback_span:
            writeback_span.update(
                output={"redis": True, "postgres": True},
                metadata={"session_key": session_key},
            )

        final_text = await self.de_anonymizer.restore(masked_response, session_key)
        if trace:
            trace.update(
                output={"source": "agent_engine"},
                metadata={"session_key": session_key, "cache_hit": False},
            )

        return {
            "source": "agent_engine",
            "response": final_text,
            "session_key": session_key,
        }
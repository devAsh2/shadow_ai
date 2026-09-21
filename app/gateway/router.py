from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
import uuid
from app.core.database import get_db_session
from app.ingress.sanitizer import SanitizerService
from app.cache.createEmbedding import EmbeddingService
from app.cache.semantic_cache import SemanticCacheService
from app.cache.pg_cache import PostgresCacheService
from app.services.proxy_service import ProxyService
from app.egress.de_anonymizer import DeAnonymizerService
from app.core.observability import langfuse_client, get_langfuse_callback

router = APIRouter(prefix="/gateway", tags=["Gateway Core"])

# Request / Response Schemas
class QueryRequest(BaseModel):
    prompt: str = Field(..., example="Contact ash@gmail.com or call 7975429427")

class QueryResponse(BaseModel):
    source: str
    response: str
    session_key: str

# Reusable Singletons / Dependencies
_embedding_service = EmbeddingService()

def get_redis_client() -> redis.Redis:
    # Uses the existing client or passes from app state
    return redis.Redis(host="localhost", port=6379, decode_responses=False)

def get_proxy_service(
    db: AsyncSession = Depends(get_db_session),
    redis_client: redis.Redis = Depends(get_redis_client),
) -> ProxyService:
    sanitizer = SanitizerService(redis_client=redis_client)
    redis_cache = SemanticCacheService(redis_client=redis_client)
    pg_cache = PostgresCacheService(session=db)
    de_anonymizer = DeAnonymizerService(redis_client=redis_client)
    
    return ProxyService(
        sanitizer=sanitizer,
        embedding_service=_embedding_service,
        redis_cache=redis_cache,
        pg_cache=pg_cache,
        de_anonymizer=de_anonymizer,
    )

@router.post("/query", response_model=QueryResponse)
async def query_gateway(
    request: QueryRequest,
    proxy_service: ProxyService = Depends(get_proxy_service),
):
    session_key = f"sess_{uuid.uuid4().hex[:12]}"

    # 1. Initialize root trace (Record session_key, keep raw PII out of top-level tags)
    trace = langfuse_client.trace(
        id=session_key,
        name="gateway_query",
        tags=["production"],
        metadata={"client": "api_gateway"}
    )

    # 2. Get callback handler to pass to LangGraph inside ProxyService
    langfuse_cb = get_langfuse_callback(trace_id=session_key)

    try:
        # 3. Delegate to ProxyService, passing session_key and tracer callback
        result = await proxy_service.handle_prompt(
            raw_prompt=request.prompt,
            session_key=session_key,
            langfuse_callback=langfuse_cb,
            trace=trace,
        )

        # 4. Finalize trace details
        trace.update(
            output=result.get("masked_response", "cached_or_completed"),
            metadata={
                "source": result["source"],
                "cache_hit": result["source"] in ("redis_cache", "postgres_cache"),
            }
        )

        return QueryResponse(
            source=result["source"],
            response=result["response"],
            session_key=result["session_key"],
        )

    except Exception as e:
        trace.update(status_message=str(e), level="ERROR")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Flushes traces asynchronously without blocking
        langfuse_client.flush()
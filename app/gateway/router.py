from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from app.core.database import get_db_session
from app.ingress.sanitizer import SanitizerService
from app.cache.createEmbedding import EmbeddingService
from app.cache.semantic_cache import SemanticCacheService
from app.cache.pg_cache import PostgresCacheService
from app.services.proxy_service import ProxyService
from app.egress.de_anonymizer import DeAnonymizerService

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
    result = await proxy_service.handle_prompt(request.prompt)
    
    # If hit in cache (Redis or Postgres), return masked answer directly
    if result["source"] in ("redis_cache", "postgres_cache"):
        return QueryResponse(
            source=result["source"],
            response=result["masked_response"],
            session_key=result["session_key"],
        )
    
    # If miss, return placeholder until LangGraph agent is wired
    return QueryResponse(
        source="miss",
        response="[Cache Miss] Will forward to LangGraph agent engine.",
        session_key=result["session_key"],
    )
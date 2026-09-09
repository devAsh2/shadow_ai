from fastapi import FastAPI
from contextlib import asynccontextmanager
import redis.asyncio as redis

from app.cache.semantic_cache import SemanticCacheService
from app.cache.pg_cache import PostgresCacheService
from app.ingress.router import router as ingress_router
from app.gateway.router import router as gateway_router  

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=False)
cache_service = SemanticCacheService(redis_client=redis_client)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[INFO] Initializing RediSearch Vector Index...")
    await cache_service.init_index()
    print("[INFO] RediSearch Vector Index ready.")

    print("[INFO] Initializing PostgreSQL pgvector tables...")
    await PostgresCacheService.init_db()
    print("[INFO] Database & Caches ready.")

    yield

    print("[INFO] Closing Redis connection...")
    await redis_client.close()

app = FastAPI(
    title="Shadow AI",
    version="0.0.1",
    description="Enterprise privacy & security gateway for LLM inference",
    debug=True,
    lifespan=lifespan,
)

# Existing debugging endpoint for sanitizer
app.include_router(ingress_router, tags=["Ingress layer"])

# New production pipeline endpoint
app.include_router(gateway_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Shadow AI Proxy"}
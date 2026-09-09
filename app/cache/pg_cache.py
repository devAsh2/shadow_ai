from typing import Optional, List
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.cache.models import SemanticCacheRecord
from app.core.database import engine, Base

class PostgresCacheService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    async def init_db():
        """Creates the pgvector extension, tables, and HNSW index."""
        async with engine.begin() as conn:
            # 1. Enable extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            # 2. Create tables
            await conn.run_sync(Base.metadata.create_all)
            # 3. Create HNSW index for sub-linear cosine distance queries
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_hnsw 
                ON semantic_cache 
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """))

    async def check_cache(self, embedding: List[float], threshold: float = 0.08) -> Optional[str]:
        """
        Executes KNN search in PostgreSQL using the cosine distance operator (<=>).
        Returns masked_response if distance <= threshold, else None.
        """
        # Distance calculation: (embedding <=> :vector)
        distance_col = SemanticCacheRecord.embedding.cosine_distance(embedding).label("distance")

        stmt = (
            select(SemanticCacheRecord.masked_response, distance_col)
            .where(distance_col <= threshold)
            .order_by(distance_col.asc())
            .limit(1)
        )

        result = await self.session.execute(stmt)
        row = result.first()
        
        if row:
            return row.masked_response
        return None

    async def set_cache(self, masked_prompt: str, masked_response: str, embedding: List[float]):
        """Persists the sanitized prompt and response long-term in PostgreSQL."""
        record = SemanticCacheRecord(
            masked_prompt=masked_prompt,
            masked_response=masked_response,
            embedding=embedding,
        )
        self.session.add(record)
        await self.session.commit()
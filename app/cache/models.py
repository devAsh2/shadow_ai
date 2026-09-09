from sqlalchemy import Column, Integer, Text, DateTime, func, text
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class SemanticCacheRecord(Base):
    __tablename__ = "semantic_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    masked_prompt = Column(Text, nullable=False)
    masked_response = Column(Text, nullable=False)
    # 384 dimensions matching all-MiniLM-L6-v2
    embedding = Column(Vector(384), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    ## need to run sqlalchemy migration to define hnsw index on embedding column
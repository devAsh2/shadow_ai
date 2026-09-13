import redis.asyncio as redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import uuid 

INDEX_NAME = "idx:semantic_cache"
DOC_PREFIX = "cache:"

class SemanticCacheService:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def init_index(self):
        """
        Creates the RediSearch vector index if it doesn't already exist.
        Idempotent: catches the error if the index already exists.
        """
        try:
            # Check if index exists by querying its metadata
            await self.redis.ft(INDEX_NAME).info()
        except Exception:
            # Index does not exist yet; define the schema
            schema = (
                TextField("masked_prompt"),
                TextField("masked_response"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 384,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            definition = IndexDefinition(prefix=[DOC_PREFIX], index_type=IndexType.HASH)
            await self.redis.ft(INDEX_NAME).create_index(fields=schema, definition=definition)

    async def check_cache(self, query_vector: bytes, threshold: float = 0.08) -> str | None:
        """
        Queries Redis for the closest vector.
        - Distance <= 0.08 means Similarity >= 0.92 (Cache Hit).
        - Returns masked_response on hit, None on miss.
        """
        # RediSearch syntax: *=>[KNN 1 @field $param AS score_name]
        query_str = "*=>[KNN 1 @embedding $vec AS vector_distance]"

        q = (
            Query(query_str)
            .sort_by("vector_distance")
            .return_fields("masked_response", "vector_distance")
            .dialect(2)
        )
        params = {"vec": query_vector}

        try:
            results = await self.redis.ft(INDEX_NAME).search(q, query_params=params)

            if results.docs:
                nearest = results.docs[0]
                distance = float(nearest.vector_distance)
                print(f"nearest distance found: {distance:.2f}")
                # Cosine distance: lower is closer. 0.08 = 92% match
                if distance <= threshold:
                    return nearest.masked_response

        except Exception:
            # If search fails (e.g. index still building or empty), treat as a miss
            return None

        return None

    async def set_cache(self, 
        masked_prompt: str, 
        masked_response: str, 
        query_vector: bytes, 
        ttl: int = 86400
    ) -> None:
        """
        Stores a new sanitized prompt-response pair with its vector embedding.
        Default TTL: 86400 seconds (24 hours).
        """
        # Generate a unique key prefixed with DOC_PREFIX ("cache:")
        key = f"{DOC_PREFIX}{uuid.uuid4().hex}"

        # Store as a Redis Hash
        await self.redis.hset(
            key,
            mapping={
                "masked_prompt": masked_prompt,
                "masked_response": masked_response,
                "embedding": query_vector,
            },
        )

        # Set key expiration (TTL)
        await self.redis.expire(key, ttl)
import asyncio
import redis.asyncio as redis
from app.cache.createEmbedding import EmbeddingService
from app.cache.semantic_cache import SemanticCacheService

async def run_test():
    client = redis.Redis(host="localhost",port=6380,decode_responses=False)
    cache = SemanticCacheService(client)
    embedder = EmbeddingService()

    print("Ensuring index exists;create the index")
    await cache.init_index()

    prompt_original = "How do I reset my account password?"
    dummy_response = "To reset your password, visit the settings security tab."
    v_bytes, _ = await embedder.get_embeddings(prompt_original)

    #Seed it into the cache 
    print("Seeding cache in redis")
    await cache.set_cache(
        masked_prompt=prompt_original,
        masked_response=dummy_response,
        query_vector=v_bytes,
        ttl=60
    )

    print("[3] Testing Exact Match Hit...")
    hit_exact = await cache.check_cache(v_bytes)
    assert hit_exact == dummy_response, f"Exact hit failed! Got : {hit_exact}"
    print("-> PASS: Exact match returned expected response.")

    print("[4] Testing Paraphrased Semantic Hit (Cosine Distance <= 0.08)...")
    prompt_paraphrase = "Where can I change my profile password?"
    p_bytes, _ = await embedder.get_embeddings(prompt_paraphrase)
    hit_paraphrase = await cache.check_cache(p_bytes,threshold=0.28)
    assert hit_paraphrase == dummy_response, f"Semantic hit failed! Got: {hit_paraphrase}"
    print("  -> PASS: Paraphrased query successfully triggered Cache Hit.")

    print("[5] Testing Unrelated Query Miss (Cosine Distance > 0.08)...")
    prompt_unrelated = "What is the capital of France?"
    u_bytes, _ = await embedder.get_embeddings(prompt_unrelated)
    miss_result = await cache.check_cache(u_bytes,threshold=0.28)
    assert miss_result is None, f"Expected None on miss, got: {miss_result}"
    print("  -> PASS: Unrelated query correctly triggered Cache Miss.")

    await client.close()
    print("\n[ALL REDIS CACHE TESTS PASSED]")

if __name__ == "__main__":
    asyncio.run(run_test())
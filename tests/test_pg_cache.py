import asyncio
from app.core.database import async_session_maker
from app.cache.pg_cache import PostgresCacheService
from app.cache.createEmbedding import EmbeddingService

async def run_test():
    print("[1] Initializing PostgreSQL Schema & Vector Index...")
    await PostgresCacheService.init_db()

    embedder = EmbeddingService()
    prompt_original = "How do I reset my account password?"
    dummy_response = "To reset your password, visit the settings security tab."

    # pgvector takes the float list, not raw bytes
    _, v_list = await embedder.get_embeddings(prompt_original)

    async with async_session_maker() as session:
        pg_service = PostgresCacheService(session)

        print("[2] Seeding Record into PostgreSQL pgvector...")
        await pg_service.set_cache(
            masked_prompt=prompt_original,
            masked_response=dummy_response,
            embedding=v_list
        )

        print("[3] Testing Exact Match Hit...")
        exact_hit = await pg_service.check_cache(v_list, threshold=0.28)
        assert exact_hit == dummy_response, f"Exact hit failed! Got: {exact_hit}"
        print("  -> PASS: Exact match returned expected response.")

        print("[4] Testing Paraphrased Semantic Hit (Threshold <= 0.28)...")
        prompt_paraphrase = "Where can I change my profile password?"
        _, p_list = await embedder.get_embeddings(prompt_paraphrase)
        
        hit_paraphrase = await pg_service.check_cache(p_list, threshold=0.28)
        assert hit_paraphrase == dummy_response, f"Semantic hit failed! Got: {hit_paraphrase}"
        print("  -> PASS: PostgreSQL pgvector retrieved record on paraphrase.")

        print("[5] Testing Unrelated Query Miss (Threshold <= 0.28)...")
        prompt_unrelated = "What is the capital of France?"
        _, u_list = await embedder.get_embeddings(prompt_unrelated)

        miss_result = await pg_service.check_cache(u_list, threshold=0.28)
        assert miss_result is None, f"Expected None on miss, got: {miss_result}"
        print("  -> PASS: Unrelated query correctly triggered Cache Miss.")

    print("\n[ALL POSTGRES PGVECTOR TESTS PASSED]")

if __name__ == "__main__":
    asyncio.run(run_test())
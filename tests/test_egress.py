import json
import asyncio
import redis.asyncio as redis
from app.egress.de_anonymizer import DeAnonymizerService


async def run_egress_test():
    # Connect to the Redis Stack port configured earlier (6380)
    client = redis.Redis(host="127.0.0.1", port=6380, decode_responses=False)
    de_anonymizer = DeAnonymizerService(client)

    session_key = "test_egress_session_42"
    redis_key = f"rev_map_session:{session_key}"

    reverse_map = {
        "<PERSON_1>": "Aditi Sharma",
        "<EMAIL_ADDRESS_1>": "aditi.sharma@example.com",
        "<PHONE_NUMBER_1>": "+91 9876543210",
    }

    print("\n--- [TEST 1: Seed Reverse Map into Redis] ---")
    await client.set(redis_key, json.dumps(reverse_map), ex=60)
    print(f"Seeded reverse mapping under key: {redis_key}")

    print("\n--- [TEST 2: Standard Multi-Entity De-Anonymization] ---")
    masked_text = (
        "Hello <PERSON_1>, we have dispatched your welcome packet to "
        "<EMAIL_ADDRESS_1> and notified your mobile <PHONE_NUMBER_1>."
    )

    restored_text = await de_anonymizer.restore(
        masked_text=masked_text,
        session_key=session_key,
    )
    print(f"Masked Input:   {masked_text}")
    print(f"Restored Text:  {restored_text}")

    assert "Aditi Sharma" in restored_text, "Failed to restore <PERSON_1>!"
    assert "aditi.sharma@example.com" in restored_text, "Failed to restore <EMAIL_ADDRESS_1>!"
    assert "+91 9876543210" in restored_text, "Failed to restore <PHONE_NUMBER_1>!"
    assert "<PERSON_1>" not in restored_text, "Placeholder token leaked into output!"
    print("-> PASS: Multi-entity restoration matches ground truth.")

    print("\n--- [TEST 3: Missing Session Key Fallback / Graceful Handling] ---")
    non_existent_session = "invalid_session_999"
    fallback_text = "Notice sent to <EMAIL_ADDRESS_1>."
    
    # Should safely return the text as-is without crashing if session is missing/expired
    unmodified_result = await de_anonymizer.restore(
        masked_text=fallback_text,
        session_key=non_existent_session,
    )
    print(f"Fallback Text on Missing Key: {unmodified_result}")
    assert unmodified_result == fallback_text, "Service crashed or modified text on missing session!"
    print("-> PASS: Gracefully handled expired/missing Redis session key.")

    # Cleanup
    await client.delete(redis_key)
    await client.aclose()
    print("\n[ALL EGRESS LAYER TESTS PASSED]")


if __name__ == "__main__":
    asyncio.run(run_egress_test())
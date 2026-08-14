import redis.asyncio as redis

#local redis store (will pass this to sanitizer to cache the pii_map)
#shared redis pool dependency 
async def get_redis():
    client = redis.from_url("redis://localhost:6379", decode_responses=True)
    try:
        yield client
    finally: 
        await client.close()
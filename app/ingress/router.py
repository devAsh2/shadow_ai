from app.ingress.sanitizer import SanitizerService
##route decorator and router function 
## traffic controller for ingress layer
##1. Contract Definition    (Pydantic Request/Response DTOs)  │
# │ 2. HTTP Routing           (Prefix, Tags, Status Codes)      │
# │ 3. Dependency Injection   (Database/Redis/Auth instances)   │
# │ 4. Service Delegation     (Pass payload -> Service Layer)   │
# │ 5. HTTP Error Translation (Catch domain exceptions -> 4xx)

from fastapi  import APIRouter,  Depends
import redis.asyncio as redis
from ..helper import get_redis
from .schema import SanitizerRequest, SanitizerResponse

router = APIRouter(prefix="/v1",tags="Ingress Governance")

router.post("/sanitize", response_model=SanitizerResponse, status_code=200)
async def sanitize_endpoint(payload: SanitizerRequest, redis_client:redis.Redis = Depends(get_redis)):
    service = SanitizerService(redis_client)
    result = await service.sanitize_and_store(payload.prompt)
    return result
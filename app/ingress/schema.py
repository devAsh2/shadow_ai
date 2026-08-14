from pydantic import BaseModel

class SanitizerRequest(BaseModel):
    prompt: str

class SanitizerResponse(BaseModel):
    session_key: str
    masked_prompt: str
    has_pii: bool
    TTL: int
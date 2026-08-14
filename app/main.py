# fastapi entry point and router registration
from fastapi import FastAPI
from app.ingress.router import router as ingress_router

app = FastAPI(
    title="Shadow AI",
    version="0.0.1",
    description="Enterprise privacy & security gateway for LLM inference",
    debug=True
)

# Register Ingress layer router
app.include_router(ingress_router, tags=["Ingress layer"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Shadow AI Proxy"}
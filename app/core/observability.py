import os
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Initialize base client
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
)

def get_langfuse_callback(trace_id: str, user_id: str = "anonymous", tags: list = None) -> CallbackHandler:
    """
    Returns a LangChain/LangGraph callback handler bound to the specific trace.
    """
    return CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        trace_id=trace_id,
        user_id=user_id,
        tags=tags or ["shadow-ai-gateway"],
    )
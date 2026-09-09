from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # add_messages automatically appends new messages rather than overwriting
    messages: Annotated[Sequence[BaseMessage], add_messages]
    masked_prompt: str
    session_key: str
    final_response: str
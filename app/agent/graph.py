import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from app.agent.state import AgentState

# System instructions ensuring LLM does not hallucinate PII
SYSTEM_PROMPT = (
    "You are a helpful, secure AI assistant inside the Shadow AI Gateway. "
    "The user's prompt may contain masked placeholders like <EMAIL_ADDRESS_1> or <PHONE_NUMBER_1>. "
    "Do NOT alter, remove, or try to decode these placeholders. Always refer to them as given."
)

def get_llm():
    # Reads GROQ_API_KEY from environment
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
    )

async def agent_node(state: AgentState) -> dict:
    """Executes the LLM node with the current prompt."""
    llm = get_llm()
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=state["masked_prompt"])
    ]
    response = await llm.ainvoke(messages)

    return {
        "final_response": response.content,
        "messages": [response]
    }


def build_agent_graph():
    """Compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add node
    workflow.add_node("agent", agent_node)

    # 2. Define entry point and termination
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)

    # 3. Compile
    return workflow.compile()

# Singleton compiled graph
agent_graph = build_agent_graph()
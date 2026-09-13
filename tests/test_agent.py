import os
import pytest
import asyncio
from app.agent.graph import agent_graph

# Ensure API key is present before running
if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY environment variable is not set. Run: $env:GROQ_API_KEY='your_key'")


async def test_agent_graph_execution():
    print("\n--- [TEST 1: Basic Agent Inference] ---")
    session_key = "test_agent_session_001"
    masked_prompt = "Hello! In one short sentence, what is 2 + 2?"

    initial_state = {
        "masked_prompt": masked_prompt,
        "session_key": session_key,
        "messages": [],
        "final_response": "",
    }

    result = await agent_graph.ainvoke(initial_state)

    print(f"Agent Raw Output: {result.get('final_response')}")
    assert "final_response" in result, "State missing 'final_response' key!"
    assert len(result["final_response"].strip()) > 0, "Agent returned an empty response!"
    print("-> PASS: LangGraph compiled and produced a valid response.")


async def test_agent_surrogate_token_retention():
    print("\n--- [TEST 2: Surrogate Token Handling] ---")
    session_key = "test_agent_session_002"
    # Prompt explicitly using surrogate tokens
    masked_prompt = (
        "Draft a formal 1-sentence confirmation: "
        "'Dear <PERSON_1>, your passkey has been sent to <EMAIL_ADDRESS_1>.'"
    )

    initial_state = {
        "masked_prompt": masked_prompt,
        "session_key": session_key,
        "messages": [],
        "final_response": "",
    }

    result = await agent_graph.ainvoke(initial_state)
    output = result.get("final_response", "")

    print(f"Agent Surrogate Output: {output}")

    # The downstream LLM must preserve the surrogate tokens for Egress to reconstruct
    has_person = "<PERSON_1>" in output or "PERSON_1" in output
    has_email = "<EMAIL_ADDRESS_1>" in output or "EMAIL_ADDRESS_1" in output

    assert has_person or has_email, (
        f"LLM stripped surrogate placeholders from the text! Output was: {output}"
    )
    print("-> PASS: Surrogate placeholders preserved in agent output.")


async def run_all():
    await test_agent_graph_execution()
    await test_agent_surrogate_token_retention()
    print("\n[ALL AGENT LAYER TESTS PASSED]")


if __name__ == "__main__":
    asyncio.run(run_all())
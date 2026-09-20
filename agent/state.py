"""
The shared state object that flows through every node of the graph.
`messages` uses LangGraph's `add_messages` reducer, which means nodes don't
overwrite the message list — they append to it (new AI messages, tool
results, etc. all accumulate), giving the agent its conversation memory
within a single request.
"""
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: Optional[str]           # "sales" | "support"  — set by classify_intent
    session_id: str
    customer_email: Optional[str]

"""
This is the required LangGraph workflow. Shape (matches the diagram in the
assessment brief, plus an off_topic guardrail branch):

                              START
                                |
                       classify_intent  (LLM: "sales" / "support" / "off_topic")
                                |
                 -----------------------------------
                 |               |                 |
            sales_agent    support_agent       off_topic
                 |   ^           |   ^               |
                 v   |           v   |               v
               tools -         tools -               END
                 |               |
                 v               v
                    ----> END <----

- classify_intent: one cheap LLM call that routes the turn. It has a third
  label, "off_topic", specifically so the agent doesn't answer things that
  have nothing to do with the store (general knowledge, recipes, etc.) —
  this is a real requirement, not just prompt-level politeness: keeping the
  agent's job scoped to the business it represents.
- off_topic: a small LLM call with NO tools bound — architecturally, this
  branch cannot call create_order/rag_search/etc even if it wanted to. It
  just produces one short declining sentence and redirects the customer back
  to store topics, then the graph ends immediately (no tool loop).
- sales_agent / support_agent: LLM calls with tools bound via bind_tools().
  If the model decides it needs a tool (e.g. search_products, rag_search,
  create_order), it emits a tool_call instead of a final answer. Both system
  prompts also reinforce the same "store topics only" scope as a second,
  softer layer — the classifier is the hard gate, the prompts are backup for
  borderline messages that got routed into a specialist anyway.
- tools: a single shared ToolNode that actually executes whichever tool(s)
  the agent requested (this is what makes RAG retrieval, the create_order
  business action, and email validation really happen, not just get
  described by the LLM).
- After tools run, control returns to the SAME agent node (sales stays with
  sales, support stays with support) so it can read the tool result and
  either call another tool or produce the final answer. This loop is capped
  implicitly by the model choosing to stop calling tools; a max-turn guard
  is added for safety.
"""
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, AIMessage

from agent.state import AgentState
from agent.llm import get_llm
from agent.prompts import (
    INTENT_SYSTEM_PROMPT, SALES_SYSTEM_PROMPT, SUPPORT_SYSTEM_PROMPT, OFF_TOPIC_SYSTEM_PROMPT,
)
from agent.tools import SALES_TOOLS, SUPPORT_TOOLS, ALL_TOOLS

MAX_TOOL_LOOPS = 4
VALID_INTENTS = {"sales", "support", "off_topic"}


def classify_intent(state: AgentState) -> dict:
    llm = get_llm(temperature=0)
    history_snippet = state["messages"][-6:]  # short context window is enough to route
    resp = llm.invoke([SystemMessage(content=INTENT_SYSTEM_PROMPT)] + history_snippet)
    label = resp.content.strip().lower()
    intent = label if label in VALID_INTENTS else "support"  # safe default if the LLM is verbose
    return {"intent": intent}


def _run_agent(state: AgentState, system_prompt: str, tools: list) -> dict:
    llm = get_llm()
    if tools:
        llm = llm.bind_tools(tools)
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    ai_msg = llm.invoke(messages)
    return {"messages": [ai_msg]}


def sales_agent(state: AgentState) -> dict:
    return _run_agent(state, SALES_SYSTEM_PROMPT, SALES_TOOLS)


def support_agent(state: AgentState) -> dict:
    return _run_agent(state, SUPPORT_SYSTEM_PROMPT, SUPPORT_TOOLS)


def off_topic_node(state: AgentState) -> dict:
    # No tools passed in at all — this branch has no way to touch the DB,
    # RAG, or place an order, by construction, not just by instruction.
    return _run_agent(state, OFF_TOPIC_SYSTEM_PROMPT, tools=[])


tool_node = ToolNode(ALL_TOOLS)


def route_after_intent(state: AgentState) -> str:
    return {"sales": "sales_agent", "support": "support_agent", "off_topic": "off_topic"}[state["intent"]]


def _tool_loop_count(state: AgentState) -> int:
    return sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls)


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    wants_tool = isinstance(last, AIMessage) and bool(last.tool_calls)
    if wants_tool and _tool_loop_count(state) <= MAX_TOOL_LOOPS:
        return "tools"
    return END


def route_after_tools(state: AgentState) -> str:
    # send control back to whichever specialist owns this turn
    return "sales_agent" if state["intent"] == "sales" else "support_agent"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("sales_agent", sales_agent)
    graph.add_node("support_agent", support_agent)
    graph.add_node("off_topic", off_topic_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent", route_after_intent,
        {"sales_agent": "sales_agent", "support_agent": "support_agent", "off_topic": "off_topic"},
    )

    graph.add_conditional_edges("sales_agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("support_agent", route_after_agent, {"tools": "tools", END: END})
    graph.add_edge("off_topic", END)  # no tools on this branch, nothing to loop on

    graph.add_conditional_edges("tools", route_after_tools,
                                 {"sales_agent": "sales_agent", "support_agent": "support_agent"})

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

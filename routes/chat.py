"""
This blueprint is the bridge between HTTP and the LangGraph agent:
1. Load (or create) the Conversation for this session_id and replay its
   history into LangChain message objects, so the agent has memory across
   turns (requirement: "maintain conversation context").
2. Append the new user message, run the graph.
3. Persist every new message the graph produced (assistant replies; tool
   activity is implicit in the final assistant message) back to the DB so
   the dashboard's "Conversations" screen can show it.
"""
import uuid
from flask import Blueprint, request, jsonify, render_template
from langchain_core.messages import HumanMessage, AIMessage

from models import db, Conversation, Message
from agent.graph import get_graph

chat_bp = Blueprint("chat", __name__)


def _load_history_as_messages(conversation: Conversation):
    msgs = []
    for m in conversation.messages:
        if m.role == "user":
            msgs.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            msgs.append(AIMessage(content=m.content))
    return msgs


@chat_bp.route("/chat")
def chat_page():
    return render_template("chat.html")


@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    customer_email = data.get("customer_email")

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    conversation = Conversation.query.filter_by(session_id=session_id).first()
    if not conversation:
        conversation = Conversation(session_id=session_id, customer_email=customer_email)
        db.session.add(conversation)
        db.session.commit()
    elif customer_email and not conversation.customer_email:
        conversation.customer_email = customer_email
        db.session.commit()

    history = _load_history_as_messages(conversation)
    history.append(HumanMessage(content=user_message))

    graph = get_graph()
    result = graph.invoke({
        "messages": history,
        "intent": None,
        "session_id": session_id,
        "customer_email": customer_email,
    })

    # The last AI message with actual text content is the reply shown to the user
    final_ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage) and m.content]
    reply = final_ai_messages[-1].content if final_ai_messages else "Sorry, I couldn't process that."
    intent = result.get("intent")

    db.session.add(Message(conversation_id=conversation.id, role="user",
                            content=user_message, intent=intent))
    db.session.add(Message(conversation_id=conversation.id, role="assistant",
                            content=reply, intent=intent))
    db.session.commit()

    return jsonify({"reply": reply, "intent": intent, "session_id": session_id})

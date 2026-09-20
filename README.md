# TechHub Electronics — AI Sales & Customer Service Agent

An AI agent for an **electronics e-commerce store** that can answer product/policy
questions using RAG, hold a real conversation, and take a real business action
(placing an order) through tool calling — built with **LangGraph**, **Flask**,
**SQLAlchemy**, and a lightweight local vector store (numpy).

---

## 1. Business domain

**TechHub Electronics** — an online store selling laptops, audio gear, monitors,
cameras and accessories. Chosen because it naturally needs both:
- **Sales**: recommend products, quote prices/stock, place orders, capture leads.
- **Customer service**: shipping/returns/warranty policy, order status, FAQs.

---

## 2. Architecture

```
Browser (chat.html)
      │  POST /api/chat {message, session_id, customer_email}
      ▼
Flask route (routes/chat.py)
      │  loads conversation history from DB → LangChain messages
      ▼
LangGraph agent (agent/graph.py)
      │
      ▼
 classify_intent  ──► "sales" / "support" / "off_topic"
      │
      ├──► sales_agent  ──tool_calls──► tools ──┐
      │        ▲                                │
      │        └────────────────────────────────┘ (loop until no more tool calls)
      │
      ├──► support_agent ──tool_calls──► tools ──┐
      │        ▲                                 │
      │        └─────────────────────────────────┘
      │
      └──► off_topic (no tools bound) ──► END directly
      │
      ▼
 final AI message ──► saved to DB (Message table) ──► returned as JSON
```

Two parallel stores of knowledge:
- **SQLite (SQLAlchemy ORM)** — structured business data: products, customers,
  orders, leads, conversations, and the *editable source* of knowledge-base docs.
- **Vector store (numpy-based, `rag/vector_store.py`)** — semantic index of the knowledge-base docs, used only
  for retrieval. `rag/rag_manager.py` is the only code that touches it, and it
  keeps the two in sync on every add/update/delete.

---

## 3. How the LangGraph agent works

File: `agent/graph.py`. State (`agent/state.py`) is a `TypedDict` with a
`messages` list (using LangGraph's `add_messages` reducer, which is what gives
the agent memory *within* a single graph run — nodes append messages rather
than overwrite them).

Nodes:
1. **`classify_intent`** — one cheap LLM call that reads the latest message (+
   a little history) and outputs `sales`, `support`, or `off_topic`. This is
   the router from the assessment's diagram, extended with a third label so
   the agent stays scoped to the business it represents (see "Staying
   on-topic" below).
2. **`sales_agent`** / **`support_agent`** — each is an LLM call with a
   *different* system prompt and a *different* set of tools bound via
   `.bind_tools()`. The model itself decides whether it needs a tool or can
   answer directly (standard function-calling / ReAct pattern).
3. **`off_topic`** — an LLM call with **no tools bound at all**. It produces
   one short sentence declining the unrelated question and redirecting to
   store topics, then goes straight to END. Because no tools are bound, this
   branch has no way to touch the database or place an order even if the
   model were somehow persuaded to try.
4. **`tools`** — a single `ToolNode` that actually executes whatever tool(s)
   the previous node's `AIMessage.tool_calls` requested (RAG search, a DB
   query, or the order-placing side-effect). After it runs, a conditional
   edge routes back to whichever specialist owns the turn, so the agent can
   read the tool's result and either call another tool or produce the final
   answer. A `MAX_TOOL_LOOPS` guard prevents infinite loops.
5. The graph ends when the latest `AIMessage` has no more tool calls — that
   message's `content` is the reply sent back to the user.

Why a graph and not one LLM call: it lets each specialist have a narrow,
focused prompt + tool set (better accuracy, easier to reason about and
extend), and it makes the tool-use loop explicit and controllable (loop
limit, per-branch tools) instead of hidden inside a single opaque call.

### Staying on-topic

Asked "what is spaghetti?", a naive agent will happily answer — it's just an
LLM with no notion of what it's *for*. Two layers stop that here:
1. **Hard gate**: `classify_intent` routes anything unrelated to TechHub
   Electronics to `off_topic`, a branch with zero tools bound, so it's
   architecturally incapable of doing anything except declining politely.
2. **Soft backup**: `SALES_SYSTEM_PROMPT` / `SUPPORT_SYSTEM_PROMPT` also state
   the scope explicitly, in case a borderline message slips into one of the
   specialists anyway (e.g. "what's a good gift for someone who cooks?" is
   store-adjacent enough to route to sales, but the prompt still tells it not
   to answer a pure recipe question if one follows).

### Email validation

Similarly, email format is enforced in **code, not just prompt instructions**
(prompts alone are not reliable enough for something with side effects). Every
tool in `agent/tools.py` that takes a `customer_email` — `create_order`,
`create_lead`, `check_order_status` — checks it against a regex
(`_EMAIL_RE` in `agent/tools.py`) before touching the database, and returns a
clear error string for the LLM to relay back ("that doesn't look like a valid
email…") rather than silently creating a customer record with garbage data.

---

## 4. How RAG works

- Knowledge lives in the `knowledge_documents` SQL table (title, category,
  content) — this is what the **admin dashboard** edits.
- `rag/rag_manager.py`'s `RAGManager` chunks each document (simple
  paragraph-aware chunking), embeds the chunks with a **free local
  sentence-transformers model** (`all-MiniLM-L6-v2`, no API key/cost — see
  `_build_embeddings()`), and stores them in a local persistent, dependency-light
  collection, tagging every chunk with `source_id` = the SQL row id.
- `add_document` / `update_document` / `delete_document` are called from the
  dashboard routes on every create/edit/delete, so the vector index is never
  stale relative to what the admin sees.
- The vector store itself (`rag/vector_store.py`) is a small numpy
  implementation — brute-force cosine similarity over stored embedding
  vectors, persisted to disk as `.npz` + a JSON sidecar. This is a deliberate
  choice over Chroma/FAISS: those ship compiled C++ extensions that need a
  C++ build toolchain to install on some machines (this bit me on Windows
  during development), while numpy has prebuilt wheels everywhere and is
  plenty fast at the scale of a store's FAQ/policy knowledge base (hundreds
  to low thousands of chunks). It exposes the same add/delete/search shape a
  real vector DB client would, so swapping in Chroma or a hosted vector DB
  later only touches this one file.
- The agent's `rag_search` tool calls `RAGManager.search(query, k=3)` and
  returns the top-3 chunks; the LLM is instructed (in `agent/prompts.py`) to
  ground policy/FAQ answers in this rather than guessing.
- Product facts (price/stock) are deliberately **not** put through RAG —
  they change constantly and need to be exact, so they're served by a direct
  `search_products` SQL tool instead. RAG is for the "soft", document-shaped
  knowledge (policies, FAQs); tools are for live, structured data. This
  split is a common and interview-worthy design decision.

---

## 5. Database structure

| Table | Purpose |
|---|---|
| `products` | catalog: name, brand, category, price, stock, description |
| `customers` | end users, identified by email |
| `orders` / `order_items` | placed orders and their line items |
| `leads` | captured sales inquiries that aren't orders yet |
| `conversations` / `messages` | chat history per session, for agent memory + dashboard visibility |
| `knowledge_documents` | RAG source-of-truth documents, mirrored into the vector store |

ORM: SQLAlchemy via Flask-SQLAlchemy. Swappable to Postgres by changing
`DATABASE_URL` in `.env` — no code changes needed.

---

## 6. Available tools / functions

Defined in `agent/tools.py`, each a `@tool`-decorated function:

| Tool | Used by | What it does |
|---|---|---|
| `search_products` | sales, support | keyword/category/price search over `products` |
| `check_product_availability` | sales | live stock check for one product |
| `rag_search` | both | semantic search over the knowledge base |
| `create_order` | sales | **real business action** — creates Customer if new, Order + OrderItems, decrements stock, commits to DB |
| `create_lead` | sales | captures an interested-but-not-buying customer |
| `check_order_status` | support | looks up order(s) by id or customer email |

`create_order` is the assessment's required "perform at least one real
business action" — it is a genuine DB write with rollback on failure (bad
product id, insufficient stock), not a scripted LLM response.

---

## 7. Running it locally 

This project defaults to **Groq** for the LLM (free, no credit card, very
fast — get a key in under a minute) and a **local, free embedding model**
for RAG (no key or account at all — it just downloads once from Hugging Face
and runs on your machine).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# get a free key at https://console.groq.com/keys and paste it into
# GROQ_API_KEY in .env  (LLM_PROVIDER and EMBEDDING_PROVIDER are already
# set to the free defaults — groq + local — so nothing else to change)

python seed_data.py             # creates sample products + indexes the KB
                                 # (first run downloads the local embedding
                                 #  model, ~80MB, then works offline)
python app.py                   # runs on http://localhost:5000
```

- Admin dashboard: `http://localhost:5000/admin`
- Chat widget (talk to the agent): `http://localhost:5000/chat`

Prefer Google instead? Set `LLM_PROVIDER=gemini` and get a free key at
https://aistudio.google.com/apikey — same free-tier idea, different provider.
Have an OpenAI or Anthropic key instead? Set `LLM_PROVIDER=openai` or
`anthropic` and fill in the matching key; everything else in the code stays
the same, because `agent/llm.py` isolates the provider choice to one file.

> **Note on this repo's origin**: I don't have internet access inside the tool
> sandbox I built this in, so I could not `pip install` LangGraph/LangChain/
> LangGraph/LangChain/sentence-transformers or actually execute an end-to-end run there.
> The code was written and reviewed carefully against the documented APIs of
> each library; please run the steps above to verify end-to-end and let me
> know if anything needs a tweak for your exact installed versions.

---

## 8. Environment variables

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `groq` (default, free), `gemini` (free), `openai`, or `anthropic` |
| `GROQ_API_KEY` / `GROQ_CHAT_MODEL` | free key from console.groq.com; default model `llama-3.3-70b-versatile` |
| `GOOGLE_API_KEY` / `GEMINI_CHAT_MODEL` | only needed if `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | only needed if you switch to those providers |
| `EMBEDDING_PROVIDER` | `local` (default, free, no key) or `openai` |
| `LOCAL_EMBEDDING_MODEL` | Hugging Face model id used for local embeddings |
| `FLASK_SECRET_KEY` | Flask session/flash secret |
| `DATABASE_URL` | SQLAlchemy connection string (defaults to local SQLite) |
| `VECTOR_STORE_DIR` | folder for the persistent vector store (numpy files) |

---

## 9. Example conversations

**Sales, with RAG + order tool:**
```
User: What laptops do you have under $1000?
Agent: [search_products] We have the Aurora X1 Laptop ($899.99, 16GB RAM,
512GB SSD) — 15 in stock. Want me to add one to an order?
User: Yes, one please. My email is jane@example.com, I'm Jane.
Agent: [check_product_availability → create_order] Order #1 confirmed for
jane@example.com. Total: $899.99.
```

**Customer service, with RAG:**
```
User: What's your return policy on headphones?
Agent: [rag_search] Most items can be returned within 30 days in original
condition. Opened earbuds/audio items can only be returned if defective,
for hygiene reasons.
```

---

## 10. Limitations & assumptions

- Single-tenant demo: no auth on the dashboard or the chat API.
- Intent classification is a single sales/support label; a message that's
  genuinely both gets routed once and the chosen specialist still has access
  to `search_products`/`rag_search` as a fallback, but it's not a perfect
  multi-intent router.
- Conversation memory is reconstructed from the DB on every request rather
  than kept in an in-memory LangGraph checkpointer — simpler to reason about
  and survives server restarts, at the cost of an extra DB read per turn.
- Chunking is simple paragraph-based; fine for FAQ-sized docs, would want a
  smarter splitter (e.g. recursive character splitter with overlap) for much
  longer documents.
- No Meta Messenger integration in this submission (documented as bonus/optional
  in the brief); the webhook would sit in `routes/` and call the same
  `get_graph()` used by `/api/chat`.

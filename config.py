"""
Central configuration. Everything that changes between environments (API keys,
model names, file paths) lives here and is read from environment variables so
no secret is ever hard-coded in the source.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"

    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL"
    ) or f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'store.db')}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- AI provider selection ---
    # "groq" (free, recommended), "gemini" (free tier), "openai", "anthropic" (paid)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_CHAT_MODEL = os.getenv("ANTHROPIC_CHAT_MODEL", "claude-3-5-haiku-latest")

    # --- Embeddings (RAG) ---
    # "local" (default): a free sentence-transformers model running on your own
    # machine — no API key, no account, no cost. "openai" is available if you'd
    # rather use OpenAI's embedding endpoint (needs OPENAI_API_KEY either way).
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
    LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", os.path.join(BASE_DIR, "instance", "vector_store"))

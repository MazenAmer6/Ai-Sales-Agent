from config import Config


def get_llm(temperature: float = 0.2):
    """Return a LangChain chat model for whichever provider is configured.
    Keeping this in one place means the rest of the agent code never imports
    a provider-specific class directly, so switching providers is a one-line
    change in .env, not a code change.

    Free options (no credit card): "groq" and "gemini" both have generous
    free tiers and are the recommended defaults for this project.
    """
    provider = Config.LLM_PROVIDER

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=Config.GROQ_CHAT_MODEL,
            api_key=Config.GROQ_API_KEY,
            temperature=temperature,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=Config.GEMINI_CHAT_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=temperature,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=Config.ANTHROPIC_CHAT_MODEL,
            api_key=Config.ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    # default: openai
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=Config.OPENAI_CHAT_MODEL,
        api_key=Config.OPENAI_API_KEY,
        temperature=temperature,
    )

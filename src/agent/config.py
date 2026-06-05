"""Settings, loaded from the environment / .env (mirrors rag-service's config)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (this file is src/agent/config.py). Used to locate the sibling
# rag-service project's Chroma store by default.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek platform (OpenAI-compatible). The agent model lives behind this
    # single value so it can be swapped (e.g. to deepseek-v4-pro) without code
    # changes (CLAUDE.md §6). Default the key to "" so the package imports with
    # no secret; the client only fails if you actually run the live loop.
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    agent_model: str = "deepseek-v4-flash"

    # web_search backend (Tavily). Key defaults to "" so the package imports with
    # no secret; web_search only needs it when actually called.
    tavily_api_key: str = ""
    web_search_max_results: int = 3

    # doc_lookup reads P2's existing Chroma store with the same Gemini embeddings
    # it was built with (CLAUDE.md §4 -- reuse the store, never rebuild a corpus).
    # Defaults point at the sibling rag-service project; override via env.
    gemini_api_key: str = ""
    embedding_model: str = "models/gemini-embedding-001"
    doc_chroma_path: str = str(_REPO_ROOT.parent / "rag-service" / "chroma_store")
    doc_collection: str = "doc_eval_corpus"
    doc_top_k: int = 4

    # LLM judge for open-ended eval tasks (CLAUDE.md §8). A DIFFERENT model family
    # from the DeepSeek agent -- reuse the Gemini integration. A generation model,
    # distinct from the embedding model above; swappable via env.
    judge_model: str = "gemini-2.0-flash"

    # code_exec sandbox (CLAUDE.md §5). Image is built from
    # docker/sandbox.Dockerfile; the rest are the per-run resource guards.
    sandbox_image: str = "agent-sandbox:latest"
    sandbox_memory: str = "256m"
    sandbox_cpus: float = 1.0
    sandbox_pids: int = 64
    sandbox_timeout: int = 10
    # code_exec runs the Docker sandbox. Managed hosting (HF Spaces) has no Docker
    # daemon, so set this false there: code_exec then returns a clean, intentional
    # message rather than a raw Docker error. Local/self-hosted leaves it true.
    sandbox_enabled: bool = True


settings = Settings()

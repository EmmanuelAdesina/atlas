from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # The .env also carries ATLAS_* keys that are read directly from the
    # environment by the components that own them (the LLM client, the
    # notifier, the supervisor), so unknown keys are ignored here rather
    # than rejected.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # API Keys
    football_data_api_key: Optional[str] = None
    odds_api_key: Optional[str] = None

    # Database
    database_url: str = "sqlite:///data/database/atlas.db"

    # Paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"

    # HTTP Client
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff: int = 1
    user_agent: str = "Atlas/0.1.0 (research data collector)"
    rate_limit_per_second: int = 5

    # Logging
    log_level: str = "INFO"

    # --- LLM supervisor / posture proposer -------------------------------
    # Any OpenAI-compatible endpoint works (OpenRouter, OpenAI, a local
    # server). The model is only used for the exception path and for
    # natural-language posture; it never prices a bet.
    boss_api_key: Optional[str] = None
    boss_base_url: str = "https://api.openai.com/v1"
    boss_model: str = "gpt-4o-mini"

    # --- SportyBet session -----------------------------------------------
    # The supervisor re-logs in when the session expires. The credentials
    # live here so they never enter a model's context.
    sportybet_phone: Optional[str] = None
    sportybet_password: Optional[str] = None


settings = Settings()

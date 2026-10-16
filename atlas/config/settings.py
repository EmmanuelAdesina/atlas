from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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


settings = Settings()

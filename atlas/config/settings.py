from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    football_data_api_key: Optional[str] = Field(None, env="FOOTBALL_DATA_API_KEY")
    odds_api_key: Optional[str] = Field(None, env="ODDS_API_KEY")
    
    # Database
    database_url: str = Field("sqlite:///data/database/atlas.db", env="DATABASE_URL")
    
    # Paths
    raw_data_path: str = Field("data/raw", env="RAW_DATA_PATH")
    processed_data_path: str = Field("data/processed", env="PROCESSED_DATA_PATH")
    
    # HTTP Client
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    max_retries: int = Field(3, env="MAX_RETRIES")
    retry_backoff: int = Field(1, env="RETRY_BACKOFF")
    user_agent: str = Field(
        "Atlas/0.1.0 (Football Data Collector)",
        env="USER_AGENT"
    )
    rate_limit_per_second: int = Field(5, env="RATE_LIMIT_PER_SECOND")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

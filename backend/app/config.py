from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cache_ttl_seconds: int = 300
    watchlist_file: str = "data/watchlist.json"
    port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()

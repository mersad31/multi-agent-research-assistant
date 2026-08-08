from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    TAVILY_API_KEY: str | None = None
    MODEL_NAME: str = "gpt-4o-mini"
    PUBLISHER_MODEL_NAME: str = "gpt-4o"
    MAX_RETRIES: int = 2
    USE_MOCK: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
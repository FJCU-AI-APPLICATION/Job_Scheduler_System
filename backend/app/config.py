from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="envs/dev.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DB_HOST: str = "database"
    DB_PORT: int = 5432
    DB_NAME: str = "scheduler"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"
    AI_SERVICE_URL: str = "http://ai-server:8001"
    DEBUG: bool = False

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()

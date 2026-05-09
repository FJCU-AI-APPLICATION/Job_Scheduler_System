from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BACKEND_URL: str = "http://backend:8000"
    REQUEST_TIMEOUT: float = 30.0


settings = Settings()

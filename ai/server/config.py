from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_DIR: str = str(Path(__file__).parent.parent / "checkpoints")
    DEFAULT_RL_CHECKPOINT: str = "best_model.zip"
    HOST: str = "0.0.0.0"
    PORT: int = 8001


settings = Settings()

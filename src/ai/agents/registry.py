import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

ALGORITHM_MAP: dict[str, type[BaseAlgorithm]] = {
    "maskable_ppo": MaskablePPO,
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}


class ModelRegistry:
    """Manages loading and caching of SB3 model checkpoints."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._cache: dict[str, BaseAlgorithm] = {}
        self._metadata: dict | None = None

    def _load_metadata(self) -> dict:
        if self._metadata is not None:
            return self._metadata

        metadata_path = self.model_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}
        return self._metadata

    def _get_algorithm_class(self) -> type[BaseAlgorithm]:
        metadata = self._load_metadata()
        algo_name = metadata.get("algorithm", "maskable_ppo")
        return ALGORITHM_MAP.get(algo_name, MaskablePPO)

    def load_model(self, checkpoint_name: str) -> BaseAlgorithm:
        if checkpoint_name in self._cache:
            return self._cache[checkpoint_name]

        stem = checkpoint_name.removesuffix(".zip")
        checkpoint_path = self.model_dir / stem

        if not checkpoint_path.with_suffix(".zip").exists() and not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        algo_cls = self._get_algorithm_class()
        model = algo_cls.load(str(checkpoint_path))
        self._cache[checkpoint_name] = model
        return model

    def list_checkpoints(self) -> list[str]:
        if not self.model_dir.exists():
            return []
        return [f.name for f in self.model_dir.glob("*.zip")]


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        from ai.core.config import settings

        _registry = ModelRegistry(settings.MODEL_DIR)
    return _registry

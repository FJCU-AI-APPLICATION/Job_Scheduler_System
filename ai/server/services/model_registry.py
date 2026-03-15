from pathlib import Path

import torch

from models.dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelRegistry:
    """Manages loading and caching of model checkpoints."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._cache: dict[str, DQN] = {}

    def load_dqn(
        self,
        checkpoint_name: str,
        state_dim: int,
        action_dim: int,
    ) -> DQN:
        if checkpoint_name in self._cache:
            return self._cache[checkpoint_name]

        checkpoint_path = self.model_dir / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = DQN(state_dim, action_dim).to(device)
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        model.eval()
        self._cache[checkpoint_name] = model
        return model

    def list_checkpoints(self) -> list[str]:
        if not self.model_dir.exists():
            return []
        return [f.name for f in self.model_dir.glob("*.pth")]


registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global registry
    if registry is None:
        from server.config import settings

        registry = ModelRegistry(settings.MODEL_DIR)
    return registry

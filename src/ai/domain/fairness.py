"""α-fairness welfare primitive and minimization-target normalization.

Generalized Mo–Walrand α-fairness welfare function unified across the
optimizers, RL reward, and service metrics. Default α=2 reduces to Jain's
fairness index for bit-identical regression with the legacy code.

Public API:
  alpha_fairness(values, alpha)          — scalar welfare (higher = more fair)
  welfare_uniform(alpha, total, n)       — welfare at uniform distribution
  aggregate_fairness(values, alpha, kind) — single entry point; kind='welfare'
                                            or 'unfairness' (1 - welfare/welfare_uniform)
  alpha_fairness_batch(rows, alpha)      — torch batched scalar
  unfairness_batch(rows, alpha)          — torch batched aggregate(kind='unfairness')
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

EPSILON = 1e-9


def _as_tensor(values, dtype=torch.float64) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(dtype=dtype)
    if isinstance(values, np.ndarray):
        return torch.as_tensor(values, dtype=dtype)
    return torch.as_tensor(list(values), dtype=dtype)


def alpha_fairness(values, alpha: float) -> float:
    """Scalar α-welfare. Higher = more fair.

    α=0  → Σ x_i
    α=1  → Σ log(max(x_i, ε))
    α=2  → (Σx)² / (n · Σx²)
    α=∞  → min(x_i)
    else → (1/(1−α)) · Σ max(x_i, ε)^(1−α)

    Conventions: empty input returns 0; all-zero values at α=2 return 1
    (degenerate "perfect fairness"); zeros at α≥1 are clamped via EPSILON.
    """
    t = _as_tensor(values)
    n = t.shape[0]
    if n == 0:
        return 0.0

    if alpha == float("inf"):
        return float(t.min())

    if alpha == 2.0:
        sum_sq = t.pow(2).sum()
        if sum_sq == 0:
            return 1.0
        return float(t.sum().pow(2) / (n * sum_sq))

    if alpha == 0.0:
        return float(t.sum())

    clamped = t.clamp(min=EPSILON)
    if alpha == 1.0:
        return float(torch.log(clamped).sum())

    exponent = 1.0 - alpha
    return float((1.0 / exponent) * clamped.pow(exponent).sum())


def welfare_uniform(alpha: float, total: float, n: int) -> float:
    """alpha_fairness at the uniform distribution [total/n] × n.

    Closed forms (used as the normalization constant in aggregate_fairness):
      α=0 → total
      α=1 → n · log(total/n)
      α=2 → 1.0  (Jain max)
      α=∞ → total/n
      else → (1/(1−α)) · n · (total/n)^(1−α)
    """
    if n == 0 or total == 0.0:
        return 0.0

    mean = total / n

    if alpha == float("inf"):
        return mean
    if alpha == 2.0:
        return 1.0
    if alpha == 0.0:
        return float(total)
    if alpha == 1.0:
        return float(n * math.log(max(mean, EPSILON)))

    exponent = 1.0 - alpha
    return float((1.0 / exponent) * n * (max(mean, EPSILON) ** exponent))


def aggregate_fairness(
    values,
    alpha: float,
    kind: Literal["welfare", "unfairness"] = "welfare",
) -> float:
    """Single entry point for the fairness primitive.

    kind='welfare'    → alpha_fairness(values, α) directly.
    kind='unfairness' → 1 − welfare/welfare_uniform. Bounded [0,1] at α∈{2, ∞};
                         can exceed 1 at α=1 in adversarial cases (documented).
                         At α=2, exactly equals 1 − jain_fairness_index(values).
    """
    if kind == "welfare":
        return alpha_fairness(values, alpha)
    if kind == "unfairness":
        t = _as_tensor(values)
        if t.shape[0] == 0:
            return 0.0
        total = float(t.sum())
        wu = welfare_uniform(alpha, total, t.shape[0])
        if wu == 0.0:
            return 0.0
        return 1.0 - alpha_fairness(values, alpha) / wu
    raise ValueError(f"Unsupported kind {kind!r}; expected 'welfare' or 'unfairness'")


# === Batched torch variants for the EvoTorch hot path ===


def alpha_fairness_batch(rows: torch.Tensor, alpha: float) -> torch.Tensor:
    """Vectorized alpha_fairness over the first dim of `rows`.

    rows: (B, n) float tensor — each row is one population member's hours.
    Returns: (B,) float tensor of welfare values.
    """
    rows = rows.to(torch.float64)
    n = rows.shape[1]
    if n == 0:
        return torch.zeros(rows.shape[0], dtype=torch.float64, device=rows.device)

    if alpha == float("inf"):
        return rows.min(dim=1).values

    if alpha == 2.0:
        sum_sq = rows.pow(2).sum(dim=1)
        sum_v = rows.sum(dim=1)
        safe = sum_sq > 0
        out = torch.ones_like(sum_v)
        out[safe] = sum_v[safe].pow(2) / (n * sum_sq[safe])
        return out

    if alpha == 0.0:
        return rows.sum(dim=1)

    clamped = rows.clamp(min=EPSILON)
    if alpha == 1.0:
        return torch.log(clamped).sum(dim=1)

    exponent = 1.0 - alpha
    return (1.0 / exponent) * clamped.pow(exponent).sum(dim=1)


def unfairness_batch(rows: torch.Tensor, alpha: float) -> torch.Tensor:
    """Vectorized aggregate_fairness(kind='unfairness') over the first dim.

    rows: (B, n) float tensor of per-individual hours.
    Returns: (B,) float tensor of unfairness in [0, 1] (may exceed 1 at α=1).
    """
    rows = rows.to(torch.float64)
    B, n = rows.shape
    if n == 0:
        return torch.zeros(B, dtype=torch.float64, device=rows.device)

    welfare = alpha_fairness_batch(rows, alpha)
    totals = rows.sum(dim=1)

    out = torch.zeros(B, dtype=torch.float64, device=rows.device)
    safe = totals > 0
    if not safe.any():
        return out

    if alpha == float("inf"):
        wu = totals / n
    elif alpha == 2.0:
        wu = torch.ones_like(totals)
    elif alpha == 0.0:
        wu = totals
    elif alpha == 1.0:
        mean = (totals / n).clamp(min=EPSILON)
        wu = n * torch.log(mean)
    else:
        exponent = 1.0 - alpha
        mean = (totals / n).clamp(min=EPSILON)
        wu = (1.0 / exponent) * n * mean.pow(exponent)

    nz_wu = (wu != 0) & safe
    out[nz_wu] = 1.0 - welfare[nz_wu] / wu[nz_wu]
    return out

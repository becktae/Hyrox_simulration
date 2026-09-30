import numpy as np

from .engine import Conditions, Strategy, simulate
from .profile import AthleteProfile


def run(profile: AthleteProfile, strategy: Strategy, n: int = 1000, seed: int | None = None,
        target: float | None = None, cond: Conditions | None = None) -> dict:
    rng = np.random.default_rng(seed)
    totals = np.array([simulate(profile, strategy, rng, cond)["total"] for _ in range(n)])
    out = {
        "n": n,
        "mean": float(totals.mean()),
        "p10": float(np.percentile(totals, 10)),
        "p50": float(np.percentile(totals, 50)),
        "p90": float(np.percentile(totals, 90)),
    }
    counts, edges = np.histogram(totals, bins=30)
    out["hist"] = {"counts": counts.tolist(), "edges": [float(e) for e in edges]}
    if target is not None:
        out["p_beat_target"] = float((totals <= target).mean())
    return out

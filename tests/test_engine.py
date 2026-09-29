import numpy as np

from sim.engine import Strategy, simulate
from sim.montecarlo import run
from sim.profile import AthleteProfile
from sim.segments import ALL, KIND, NAMES


def fake_profile(noise=0.0):
    base = {n: (270.0 if KIND[n] == "run" else 220.0) for n in NAMES}
    base["roxzone"] = 400.0
    return AthleteProfile("T", "XXX", "M", "open", "season-8", 3, base,
                          {n: noise for n in ALL}, noise, True)


def test_plan_pace_matches_base_without_noise():
    r = simulate(fake_profile(), Strategy(), np.random.default_rng(0))
    assert abs(r["total"] - fake_profile().expected_total) < 1e-6


def test_early_aggression_costs_later():
    rng = np.random.default_rng(0)
    plan = simulate(fake_profile(), Strategy(), rng)["splits"]
    aggr = simulate(fake_profile(), Strategy(run_pace={"run1": "aggressive", "run2": "aggressive"}), rng)["splits"]
    assert aggr["run1"] < plan["run1"]           # 당장은 이득
    late = ["farmers", "sandbag", "wall_balls"]  # 후반엔 손해
    assert sum(aggr[s] for s in late) > sum(plan[s] for s in late)


def test_montecarlo_distribution_ordered():
    m = run(fake_profile(0.03), Strategy(), n=200, seed=1)
    assert m["p10"] <= m["p50"] <= m["p90"]

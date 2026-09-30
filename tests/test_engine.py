import numpy as np

from sim.engine import Conditions, Strategy, simulate
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


def _total(strategy=None, cond=None):
    return simulate(fake_profile(), strategy or Strategy(), np.random.default_rng(0), cond)["total"]


def test_neutral_conditions_match_base():
    assert abs(_total(cond=Conditions()) - fake_profile().expected_total) < 1e-6


def test_conditions_direction():
    base = _total()
    assert _total(cond=Conditions(condition=90)) < base < _total(cond=Conditions(condition=10))
    assert _total(cond=Conditions(temperature=32, humidity=80)) > base
    assert _total(cond=Conditions(stamina=70)) > base
    assert _total(cond=Conditions(grip=70)) > base


def test_sweat_hurts_grip_and_chalk_helps_late_grip():
    hot = Conditions(temperature=30, humidity=70)
    no = simulate(fake_profile(), Strategy(), np.random.default_rng(0), hot)
    ch = simulate(fake_profile(), Strategy(chalk={"farmers", "sandbag"}), np.random.default_rng(0), hot)
    assert no["trace"][-1]["sweat"] > simulate(fake_profile(), Strategy(), np.random.default_rng(0))["trace"][-1]["sweat"]
    assert ch["grip_end"] > no["grip_end"]


def test_station_strategy_tradeoff():
    push = simulate(fake_profile(), Strategy(station={"sled_push": "push"}), np.random.default_rng(0))["splits"]
    plan = simulate(fake_profile(), Strategy(), np.random.default_rng(0))["splits"]
    assert push["sled_push"] < plan["sled_push"]
    assert sum(push[s] for s in ["farmers", "sandbag", "wall_balls"]) > sum(plan[s] for s in ["farmers", "sandbag", "wall_balls"])

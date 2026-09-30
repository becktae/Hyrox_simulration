import numpy as np

from sim import scout
from sim.engine import Strategy, plan_reserve, simulate
from sim.profile import AthleteProfile
from sim.segments import ALL, KIND, NAMES


def fake_profile(traits=None):
    base = {n: (270.0 if KIND[n] == "run" else 220.0) for n in NAMES}
    base["roxzone"] = 400.0
    return AthleteProfile("T", "XXX", "M", "open", "season-8", 3, base, {n: 0.0 for n in ALL}, 0.0, True,
                          traits=traits or {})


def test_traits_keep_plan_pace_equal_to_base():
    for traits in ({"stamina": 90, "grip": 20}, {"stamina": 10, "grip": 95}):
        r = simulate(fake_profile(traits), Strategy(), np.random.default_rng(0))
        assert abs(r["total"] - fake_profile().expected_total) < 1e-6


def test_fitter_athlete_pays_less_for_aggression():
    aggr = Strategy(run_pace={f"run{i}": "aggressive" for i in range(1, 9)})
    fit = simulate(fake_profile({"stamina": 90, "grip": 90}), aggr, np.random.default_rng(0))["total"]
    weak = simulate(fake_profile({"stamina": 10, "grip": 10}), aggr, np.random.default_rng(0))["total"]
    assert fit < weak


def test_reserve_higher_for_high_stamina():
    assert plan_reserve({"stamina": 90})["stamina"] > plan_reserve({"stamina": 10})["stamina"]


def test_kalman_pulls_toward_observation_and_infers_missing():
    n = len(ALL)
    m, P = np.zeros(n), 0.01 * (0.5 * np.eye(n) + 0.5)      # 성분 간 상관 0.5
    obs = [(scout._vec({"rowing": 1.0}), -0.2, 0.05)]
    m2, P2 = scout.kalman(m, P, obs)
    assert m2[scout.IDX["rowing"]] < -0.1                      # 관측 쪽으로
    assert m2[scout.IDX["run1"]] < 0                           # 상관 때문에 측정 안 한 성분도 같이 이동
    assert P2[scout.IDX["rowing"], scout.IDX["rowing"]] < P[scout.IDX["rowing"], scout.IDX["rowing"]]


def test_scored_wod_direction_and_weight_required():
    n = len(ALL)
    mu, cov = np.zeros(n), 0.01 * np.eye(n)
    strong, _, _ = scout.observations({"deadlift": 240}, "M", 80, mu, cov)   # 3.0 × 체중
    weak, _, _ = scout.observations({"deadlift": 80}, "M", 80, mu, cov)      # 1.0 × 체중
    assert strong[0][1] < 0 < weak[0][1]                                      # 강할수록 시간 짧게
    _, _, skipped = scout.observations({"deadlift": 200}, "M", None, mu, cov)
    assert skipped


def test_wod_only_profile_and_wod_effect():
    slow, rep = scout.build(None, None, {"gender": "M", "race_type": "open", "age": 35, "weight_kg": 80}, {"run5k": 1800})
    fast, _ = scout.build(None, None, {"gender": "M", "race_type": "open", "age": 35, "weight_kg": 80}, {"run5k": 1200})
    assert fast.base["run3"] < slow.base["run3"]
    assert fast.n_races == 0 and rep["group"]["n"] > 100
    assert all(1 <= r["value"] <= 99 for r in rep["ratings"])

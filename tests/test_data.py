import pytest

from sim import data
from sim.config import DB_PATH
from sim.model import ConditionModel
from sim.profile import build_profile

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="hyrox.db 없음")


def test_race_type():
    assert data.race_type("season-8__HPRO_LR3") == "pro"
    assert data.race_type("season-8__HD_LR3") is None  # 더블은 제외


def test_search_groups_by_name():
    rows = data.search_athletes("Tvrdik, Tomas")
    assert rows and rows[0]["races"] >= 3


def test_load_singles_components_sum_to_total():
    rows = data.load_singles("Tvrdik, Tomas", "CZE")
    assert rows and all(abs(r["x"].sum() - r["total"]) < 1e-6 for r in rows)


def test_model_fit_recovers_condition_effect():
    """합성 데이터: 조건 B가 A보다 정확히 10% 느리면 beta 차이로 복원돼야 한다."""
    import numpy as np
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        base = rng.uniform(150, 300, 17)
        for c, f in (("season-1", 1.0), ("season-2", 1.1)):
            rows.append(dict(person=(str(i), "X"), race=f"{c}__H_{i}", season=c, rt="open", gender="M",
                             x=base * f * rng.lognormal(0, 0.02, 17), total=0))
    m = ConditionModel.fit(rows)
    d = m.beta[("season-2", "open", "M")] - m.beta[("season-1", "open", "M")]
    assert np.allclose(np.exp(d), 1.1, atol=0.01)


def test_profile_real_athlete():
    p = build_profile("Tvrdik, Tomas", "CZE")
    assert 2400 < p.expected_total < 5400 and p.n_races >= 1

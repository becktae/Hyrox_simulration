import numpy as np

from sim.model import get_model, one_factor


def test_one_factor_recovers_loadings_and_no_zero_cv():
    rng = np.random.default_rng(0)
    lam = np.array([0.02, 0.06, 0.09, 0.05])      # 첫 성분은 컨디션 영향이 거의 없는 구간(SkiErg 같은)
    psi_sd = np.array([0.035, 0.10, 0.12, 0.08])
    f = rng.standard_normal((40000, 1))
    R = f * lam + rng.standard_normal((40000, 4)) * psi_sd
    est_lam, est_psi = one_factor(np.cov(R.T, bias=True), R.var(0))
    assert np.allclose(est_lam, lam, atol=0.01)
    assert np.allclose(np.sqrt(est_psi), psi_sd, atol=0.01)
    assert np.sqrt(est_psi[0]) > 0.02                # 과거 모형은 이 값을 0.001로 눌렀다


def test_model_has_positive_specific_variation_everywhere():
    m = get_model()
    assert (m.seg_cv > 0.02).all()                   # 스키·로잉도 0.001이 아니어야 함
    assert abs(m.loads.mean() - 1) < 1e-9 and m.loads.min() > 0

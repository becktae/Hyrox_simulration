import numpy as np

from sim.model import factor_model, get_model


def test_factor_model_recovers_two_factor_structure():
    rng = np.random.default_rng(0)
    L = np.array([[0.02, 0.00], [0.08, 0.01], [0.09, 0.02], [0.01, 0.10], [0.02, 0.09], [0.00, 0.03]])
    psi_sd = np.array([0.035, 0.05, 0.06, 0.08, 0.07, 0.04])
    R = rng.standard_normal((60000, 2)) @ L.T + rng.standard_normal((60000, 6)) * psi_sd
    est, psi = factor_model(np.cov(R.T, bias=True), R.var(0), k=2)
    C = np.cov(R.T, bias=True)
    assert np.abs(est @ est.T + np.diag(psi) - C).max() < 2e-4     # 공분산 재현 (회전 무관)
    assert np.sqrt(psi).min() >= 0.02 - 1e-9 and np.sqrt(psi[0]) > 0.02
    grp = np.abs(est).argmax(1)                                    # 요인 순서·회전은 임의 → 묶음만 확인
    assert len(set(grp[1:3])) == 1 and len(set(grp[3:5])) == 1 and grp[1] != grp[3]


def test_model_has_positive_specific_variation_everywhere():
    m = get_model()
    assert (m.seg_cv > 0.02).all()                   # 스키·로잉도 0.001이 아니어야 함
    assert m.loads.shape[1] == 2 and abs((m.loads**2).sum(1).mean() - 1) < 1e-9

"""조건 효과 모델.

log x[사람, 조건, 성분] = alpha[사람, 성분] + beta[조건, 성분] + gamma[경험, 성분] + 잔차
  조건 = (시즌, 종목, 성별), 성분 = 16구간 + roxzone

같은 사람이 여러 조건에서 뛴 기록으로 beta를 추정하므로, 시즌별 규칙/무게 변화와
Pro↔Open 전환을 참가자 구성 변화와 분리해서 보정할 수 있다. 경험 항(gamma)은 N번째 경기(1, 2, 3+)에 따른
학습 효과다. 잔차로 경기 간 변동
(구간별 cv + 그날 컨디션 day_sd)을 데이터에서 추정한다.
"""
import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from . import data
from .config import CACHE_DIR
from .segments import ALL

N = len(ALL)
MODEL_VERSION = 4   # 모델 구조가 바뀌면 올린다 → 오래된 캐시는 자동 재적합
K_MAX = 3   # 경험 단계: 1번째, 2번째, 3번째 이상
Cond = tuple[str, str, str]  # (season, race_type, gender)


def cond_of(r: dict) -> Cond:
    return (r["season"], r["rt"], r["gender"])


N_FACTORS = 2       # 그날 컨디션 요인 수 (잔차 구조: 러닝 계열 vs 스테이션 계열)
PSI_FLOOR = 4e-4    # 구간 고유 변동 하한(sd 0.02): 어떤 구간도 '완전히 결정적'으로 눌리지 않게


def varimax(L: np.ndarray, iters: int = 100) -> np.ndarray:
    """요인 회전 — 적재량이 한 요인에 몰리게 해 요인을 해석하기 쉽게 한다(공분산 구조는 그대로)."""
    n, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(iters):
        Lr = L @ R
        u, sv, vt = np.linalg.svd(L.T @ (Lr**3 - Lr @ np.diag((Lr**2).sum(0)) / n))
        R = u @ vt
        if sv.sum() < d * (1 + 1e-8):
            break
        d = sv.sum()
    return L @ R


def factor_model(C: np.ndarray, var: np.ndarray, k: int = N_FACTORS, iters: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """잔차 공분산 C → k요인 모형 C ≈ ΛΛᵀ + diag(ψ) (주축 요인법 + varimax).

    Λ[c, j] = 그날 컨디션 요인 j가 구간 c에 미치는 정도(로그 단위), ψ[c] = 구간 고유 변동.
    '모든 구간에 같은 컨디션 효과(λ=1, 요인 1개)'를 가정하면 실제로 일정한 SkiErg·로잉(잔차 sd 4%)의 고유 변동이
    0으로 눌리고(cv=0.001) 컨디션 영향이 큰 스테이션은 과소평가된다. 요인 2개는 러닝 컨디션과 근력·스테이션 컨디션을 분리한다."""
    psi = 0.5 * var
    for _ in range(iters):
        w, V = np.linalg.eigh(C - np.diag(psi))
        top = np.argsort(w)[::-1][:k]
        L = V[:, top] * np.sqrt(np.maximum(w[top], 1e-12))
        new = np.maximum(var - (L**2).sum(1), PSI_FLOOR)
        if np.allclose(new, psi, atol=1e-10):
            break
        psi = new
    comm = (L**2).sum(1)
    over = comm > var - PSI_FLOOR                    # 고유 변동이 하한 아래로 내려가는 구간(Heywood)은 적재량을 줄인다
    L[over] *= np.sqrt(np.maximum(var[over] - PSI_FLOOR, 1e-12) / comm[over])[:, None]
    L = varimax(L)
    L = L[:, np.argsort(-(L**2).sum(0))]              # 설명하는 변동이 큰 요인부터
    L = L * np.where(L.sum(0) >= 0, 1.0, -1.0)
    return L, np.maximum(var - (L**2).sum(1), PSI_FLOOR)


@dataclass
class ConditionModel:
    beta: dict[Cond, np.ndarray]
    gamma: np.ndarray    # (K_MAX, N) 경험 효과, gamma[0]=0 기준
    seg_cv: np.ndarray   # 구간별 경기 간 고유 변동(로그 표준편차, 그날 컨디션 요인 제외)
    day_sd: float        # '그날 컨디션' 요인의 로그 표준편차 (구간 적재량 평균 기준 척도)
    seasons: list[str]
    loads: np.ndarray | None = None   # (N, N_FACTORS) 구간별 컨디션 요인 적재량 ÷ day_sd. 스키·로잉은 작고 스테이션은 큼

    @classmethod
    def fit(cls, rows: list[dict], iters: int = 20, min_rows: int = 30) -> "ConditionModel":
        by_p = defaultdict(list)
        for r in rows:
            by_p[r["person"]].append(r)
        people = [sorted(v, key=lambda r: r["race"]) for v in by_p.values() if len(v) >= 2]
        conds = sorted({cond_of(r) for v in people for r in v})
        cid = {c: i for i, c in enumerate(conds)}
        P = [(np.log([r["x"] for r in v]), np.array([cid[cond_of(r)] for r in v]),
              np.minimum(np.arange(len(v)), K_MAX - 1)) for v in people]
        cnt = np.zeros(len(conds))
        cnt_k = np.zeros(K_MAX)
        for _, ci, ki in P:
            np.add.at(cnt, ci, 1)
            np.add.at(cnt_k, ki, 1)
        beta = np.zeros((len(conds), N))
        gamma = np.zeros((K_MAX, N))
        for _ in range(iters):
            num = np.zeros_like(beta)
            for Y, ci, ki in P:
                np.add.at(num, ci, Y - gamma[ki] - (Y - beta[ci] - gamma[ki]).mean(0))
            new = num / np.maximum(cnt, 1)[:, None]
            beta = new - new[cnt.argmax()]
            numg = np.zeros_like(gamma)
            for Y, ci, ki in P:
                np.add.at(numg, ki, Y - beta[ci] - (Y - beta[ci] - gamma[ki]).mean(0))
            gamma = numg / np.maximum(cnt_k, 1)[:, None]
            gamma = gamma - gamma[0]
        # 잔차 분산(자유도 보정) → 구간 cv, day_sd
        res = []
        for Y, ci, ki in P:
            z = Y - beta[ci] - gamma[ki]
            res.append((z - z.mean(0)) * np.sqrt(len(Y) / (len(Y) - 1)))
        R = np.vstack(res)
        var_s = R.var(0)
        L, psi = factor_model(np.cov(R.T, bias=True), var_s)
        day_sd = float(np.sqrt((L**2).sum(1).mean()))      # 구간 평균 컨디션 변동(로그 sd)
        loads, seg_cv = L / day_sd, np.sqrt(psi)
        keep = {c: beta[i] for c, i in cid.items() if cnt[i] >= min_rows}
        seasons = sorted({c[0] for c in keep})
        return cls(keep, gamma, seg_cv, day_sd, seasons, loads)

    def to_json(self) -> str:
        return json.dumps({"version": MODEL_VERSION, "beta": {"|".join(k): v.tolist() for k, v in self.beta.items()},
                           "gamma": self.gamma.tolist(),
                           "seg_cv": self.seg_cv.tolist(), "day_sd": self.day_sd, "seasons": self.seasons,
                           "loads": self.loads.tolist()})

    @classmethod
    def from_json(cls, s: str) -> "ConditionModel":
        d = json.loads(s)
        if d.get("version") != MODEL_VERSION:
            raise ValueError("stale model cache")
        return cls({tuple(k.split("|")): np.array(v) for k, v in d["beta"].items()},
                   np.array(d["gamma"]), np.array(d["seg_cv"]), d["day_sd"], d["seasons"], np.array(d["loads"]))


_MODEL: ConditionModel | None = None


def get_model(refit: bool = False) -> ConditionModel:
    """cache/model.json 이 DB보다 새로우면 재사용, 아니면 전체 데이터로 재적합."""
    global _MODEL
    from .config import DB_PATH
    path = CACHE_DIR / "model.json"
    if not refit and _MODEL is not None:
        return _MODEL
    if not refit and path.exists() and path.stat().st_mtime >= DB_PATH.stat().st_mtime:
        try:
            _MODEL = ConditionModel.from_json(path.read_text())
            return _MODEL
        except (ValueError, KeyError):
            pass  # 구버전/손상 캐시 → 재적합
    _MODEL = ConditionModel.fit(data.load_singles())
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(_MODEL.to_json())
    return _MODEL

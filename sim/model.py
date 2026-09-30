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
MODEL_VERSION = 3   # 모델 구조가 바뀌면 올린다 → 오래된 캐시는 자동 재적합
K_MAX = 3   # 경험 단계: 1번째, 2번째, 3번째 이상
Cond = tuple[str, str, str]  # (season, race_type, gender)


def cond_of(r: dict) -> Cond:
    return (r["season"], r["rt"], r["gender"])


def one_factor(C: np.ndarray, var: np.ndarray, iters: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """잔차 공분산 C → 1요인 모형 C ≈ λλᵀ + diag(ψ) (주축 요인법).

    λ[c] = 그날 컨디션이 구간 c에 미치는 정도, ψ[c] = 구간 고유 변동.
    '모든 구간에 같은 컨디션 효과(λ=1)'를 가정하면 실제로 일정한 SkiErg·로잉(잔차 sd 4%)의 고유 변동이
    0으로 눌리고(cv=0.001) 컨디션 영향이 큰 스테이션은 과소평가된다."""
    psi = 0.5 * var
    for _ in range(iters):
        w, V = np.linalg.eigh(C - np.diag(psi))
        lam = V[:, -1] * np.sqrt(max(w[-1], 1e-12))
        lam = lam * (1 if lam.sum() >= 0 else -1)
        new = np.maximum(var - lam**2, 1e-5)
        if np.allclose(new, psi, atol=1e-10):
            break
        psi = new
    return np.maximum(lam, 1e-3), psi


@dataclass
class ConditionModel:
    beta: dict[Cond, np.ndarray]
    gamma: np.ndarray    # (K_MAX, N) 경험 효과, gamma[0]=0 기준
    seg_cv: np.ndarray   # 구간별 경기 간 고유 변동(로그 표준편차, 그날 컨디션 요인 제외)
    day_sd: float        # '그날 컨디션' 요인의 로그 표준편차 (구간 적재량 평균 기준 척도)
    seasons: list[str]
    loads: np.ndarray | None = None   # 구간별 컨디션 요인 적재량(평균 1). 스키·로잉은 작고 파머스·월볼은 큼

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
        lam, psi = one_factor(np.cov(R.T, bias=True), var_s)
        day_sd, loads = float(lam.mean()), lam / lam.mean()
        seg_cv = np.sqrt(psi)
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

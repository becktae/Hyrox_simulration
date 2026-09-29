"""조건 효과 모델.

log x[사람, 조건, 성분] = alpha[사람, 성분] + beta[조건, 성분] + 잔차
  조건 = (시즌, 종목, 성별), 성분 = 16구간 + roxzone

같은 사람이 여러 조건에서 뛴 기록으로 beta를 추정하므로, 시즌별 규칙/무게 변화와
Pro↔Open 전환을 참가자 구성 변화와 분리해서 보정할 수 있다. 잔차로 경기 간 변동
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
Cond = tuple[str, str, str]  # (season, race_type, gender)


def cond_of(r: dict) -> Cond:
    return (r["season"], r["rt"], r["gender"])


@dataclass
class ConditionModel:
    beta: dict[Cond, np.ndarray]
    seg_cv: np.ndarray   # 구간별 경기 간 변동(로그 표준편차, day 제외)
    day_sd: float        # 전 구간 공통 '그날 컨디션' 로그 표준편차
    seasons: list[str]

    @classmethod
    def fit(cls, rows: list[dict], iters: int = 20, min_rows: int = 30) -> "ConditionModel":
        by_p = defaultdict(list)
        for r in rows:
            by_p[r["person"]].append(r)
        people = [v for v in by_p.values() if len(v) >= 2]
        conds = sorted({cond_of(r) for v in people for r in v})
        cid = {c: i for i, c in enumerate(conds)}
        P = [(np.log([r["x"] for r in v]), np.array([cid[cond_of(r)] for r in v])) for v in people]
        cnt = np.zeros(len(conds))
        for _, ci in P:
            np.add.at(cnt, ci, 1)
        beta = np.zeros((len(conds), N))
        for _ in range(iters):
            num = np.zeros_like(beta)
            for Y, ci in P:
                np.add.at(num, ci, Y - (Y - beta[ci]).mean(0))
            new = num / np.maximum(cnt, 1)[:, None]
            beta = new - new[cnt.argmax()]
        # 잔차 분산(자유도 보정) → 구간 cv, day_sd
        res = []
        for Y, ci in P:
            r = (Y - beta[ci]) - (Y - beta[ci]).mean(0)
            res.append(r * np.sqrt(len(Y) / (len(Y) - 1)))
        R = np.vstack(res)
        var_s = R.var(0)
        day_var = max(R.mean(1).var() - var_s.mean() / N, 1e-6)
        seg_cv = np.sqrt(np.maximum(var_s - day_var, 1e-6))
        keep = {c: beta[i] for c, i in cid.items() if cnt[i] >= min_rows}
        seasons = sorted({c[0] for c in keep})
        return cls(keep, seg_cv, float(np.sqrt(day_var)), seasons)

    def to_json(self) -> str:
        return json.dumps({"beta": {"|".join(k): v.tolist() for k, v in self.beta.items()},
                           "seg_cv": self.seg_cv.tolist(), "day_sd": self.day_sd, "seasons": self.seasons})

    @classmethod
    def from_json(cls, s: str) -> "ConditionModel":
        d = json.loads(s)
        return cls({tuple(k.split("|")): np.array(v) for k, v in d["beta"].items()},
                   np.array(d["seg_cv"]), d["day_sd"], d["seasons"])


_MODEL: ConditionModel | None = None


def get_model(refit: bool = False) -> ConditionModel:
    """cache/model.json 이 DB보다 새로우면 재사용, 아니면 전체 데이터로 재적합."""
    global _MODEL
    from .config import DB_PATH
    path = CACHE_DIR / "model.json"
    if not refit and _MODEL is not None:
        return _MODEL
    if not refit and path.exists() and path.stat().st_mtime >= DB_PATH.stat().st_mtime:
        _MODEL = ConditionModel.from_json(path.read_text())
        return _MODEL
    _MODEL = ConditionModel.fit(data.load_singles())
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(_MODEL.to_json())
    return _MODEL

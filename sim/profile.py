"""선수 프로필: 본인의 과거 온전한 경기들을 조건 보정해 목표 조건 수준으로 환산한 평균."""
from dataclasses import dataclass

import numpy as np

from . import data
from .model import ConditionModel, cond_of, get_model
from .segments import ALL


@dataclass
class AthleteProfile:
    name: str
    nationality: str
    gender: str
    race_type: str
    season: str             # 예측 대상 시즌(조건)
    n_races: int            # 사용한 온전한 경기 수
    base: dict[str, float]  # 성분별 기준 시간(초). 16구간 + roxzone
    cv: dict[str, float]    # 성분별 경기 간 변동 (로그 sd, 추정 불확실성 반영)
    day_sd: float           # 그날 컨디션 (전 구간 공통)
    adjusted: bool          # 조건 보정을 적용했는지

    @property
    def expected_total(self) -> float:
        return sum(self.base.values())


def build_profile(name: str, nationality: str, season: str | None = None, race_type: str | None = None,
                  model: ConditionModel | None = None, hist: list[dict] | None = None) -> AthleteProfile:
    """hist: 백테스트용으로 과거 경기 목록을 직접 넘길 수 있다(시간순)."""
    model = model or get_model()
    hist = hist if hist is not None else sorted(data.load_singles(name, nationality), key=lambda r: r["race"])
    if not hist:
        raise ValueError(f"구간 기록이 온전한 싱글 경기가 없음: {name} ({nationality})")
    last = hist[-1]
    season = season or model.seasons[-1]
    rt = race_type or last["rt"]
    target = (season, rt, last["gender"])
    usable = [r for r in hist if cond_of(r) in model.beta]
    adjusted = bool(usable) and target in model.beta
    if adjusted:
        y = np.mean([np.log(r["x"]) - model.beta[cond_of(r)] for r in usable], axis=0) + model.beta[target]
        n = len(usable)
    else:
        y = np.mean([np.log(r["x"]) for r in hist], axis=0)
        n = len(hist)
    infl = np.sqrt(1 + 1 / n)          # 본인 수준 추정 불확실성 (기록이 적을수록 넓게)
    return AthleteProfile(
        name, nationality, last["gender"], rt, season, n,
        {s: float(np.exp(v)) for s, v in zip(ALL, y)},
        {s: float(v * infl) for s, v in zip(ALL, model.seg_cv)},
        float(model.day_sd * infl), adjusted)

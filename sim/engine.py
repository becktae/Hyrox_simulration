"""레이스 엔진.

구간 시간 = 기본 × 페이스 계수 × 피로 보정 × 그립 보정 × 랜덤(컨디션·구간)

피로는 비선형(deficit²)이며 '계획 페이스로 뛴 궤적' 대비 초과분만 보정한다.
→ 계획 페이스의 기대값은 개인 기준 기록(profile.base)과 일치하고,
  초반 공격은 당장 이득이 작고 후반에 페널티가 크다.
계수(PACE, FATIGUE_*, GRIP_*)는 scripts/backtest.py로 튜닝 대상.
"""
from dataclasses import dataclass, field

import numpy as np

from .profile import AthleteProfile
from .segments import GRIP_SEGMENTS, KIND, NAMES, OVERHEAD

# 러닝 페이스 선택: (시간 계수, 스태미나 소모 계수)
PACE = {"conservative": (1.03, 0.75), "plan": (1.00, 1.00), "aggressive": (0.97, 1.35)}
STAMINA_DRAIN = {"run": 0.035, "station": 0.045}
FATIGUE_GAIN = 0.35      # deficit² → 시간 증가율
GRIP_DRAIN = 0.10        # 그립 구간마다 소모
GRIP_STAMINA_COUPLING = 1.0   # 스태미나 결손이 클수록 그립도 더 털림
GRIP_GAIN = 0.25


@dataclass
class Strategy:
    """run별 페이스 선택. 미지정 구간은 default."""
    default: str = "plan"
    run_pace: dict[str, str] = field(default_factory=dict)

    def pace_of(self, seg: str) -> str:
        return self.run_pace.get(seg, self.default)


def _step(stamina: float, grip: float, seg: str, drain_mult: float) -> tuple[float, float]:
    kind = KIND[seg]
    stamina = max(0.0, stamina - STAMINA_DRAIN[kind] * drain_mult)
    if seg in GRIP_SEGMENTS:
        grip = max(0.0, grip - GRIP_DRAIN * (1 + GRIP_STAMINA_COUPLING * (1 - stamina)))
    return stamina, grip


def _plan_trajectory() -> list[tuple[float, float]]:
    """계획 페이스 시 구간 종료 시점의 (스태미나 결손, 그립 결손)."""
    stamina, grip, out = 1.0, 1.0, []
    for seg in NAMES:
        stamina, grip = _step(stamina, grip, seg, 1.0)
        out.append((1 - stamina, 1 - grip))
    return out


_PLAN = _plan_trajectory()


def simulate(profile: AthleteProfile, strategy: Strategy, rng: np.random.Generator) -> dict:
    stamina, grip = 1.0, 1.0
    day = rng.lognormal(0.0, profile.day_sd)
    splits, total = {}, 0.0
    for i, seg in enumerate(NAMES):
        t_mult, drain_mult = PACE[strategy.pace_of(seg)] if KIND[seg] == "run" else (1.0, 1.0)
        stamina, grip = _step(stamina, grip, seg, drain_mult)
        plan_s, plan_g = _PLAN[i]
        fatigue = 1.0 + FATIGUE_GAIN * ((1 - stamina) ** 2 - plan_s**2)
        grip_pen = 1.0 + (GRIP_GAIN * ((1 - grip) ** 2 - plan_g**2) if seg in GRIP_SEGMENTS else 0.0)
        t = profile.base[seg] * t_mult * fatigue * grip_pen * day * rng.lognormal(0.0, profile.cv[seg])
        splits[seg] = t
        total += t
    # Roxzone/전환 시간: 페이스 전략과 무관하게 개인 기준 + 변동 (total_time = 16구간 합 + roxzone)
    rox = profile.base[OVERHEAD] * day * rng.lognormal(0.0, profile.cv[OVERHEAD])
    splits[OVERHEAD] = rox
    total += rox
    return {"splits": splits, "total": total, "stamina_end": stamina, "grip_end": grip}

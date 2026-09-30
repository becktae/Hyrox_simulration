"""레이스 엔진.

구간 시간 = 기본 × 페이스 계수 × 피로 보정 × 그립 보정 × 랜덤(컨디션·구간)

피로는 비선형(deficit²)이며 '계획 페이스로 뛴 궤적' 대비 초과분만 보정한다.
→ 계획 페이스의 기대값은 개인 기준 기록(profile.base)과 일치하고,
  초반 공격은 당장 이득이 작고 후반에 페널티가 크다.
계수(PACE, FATIGUE_*, GRIP_*)는 scripts/backtest.py로 튜닝 대상.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .profile import AthleteProfile
from .segments import GRIP_SEGMENTS, KIND, NAMES, OVERHEAD

# 러닝 페이스 선택: (시간 계수, 스태미나 소모 계수)
PACE = {"conservative": (1.03, 0.75), "plan": (1.00, 1.00), "aggressive": (0.97, 1.35)}
# 스테이션 전략: (시간 계수, 스태미나 소모 계수, 그립 소모 계수) — 계획(steady)이 기준
STATION = {"safe": (1.04, 0.70, 0.70), "steady": (1.00, 1.00, 1.00), "push": (0.96, 1.40, 1.30)}
STAMINA_DRAIN = {"run": 0.035, "station": 0.045}
FATIGUE_GAIN = 0.35      # deficit² → 시간 증가율
GRIP_DRAIN = 0.10        # 그립 구간마다 소모
GRIP_STAMINA_COUPLING = 1.0   # 스태미나 결손이 클수록 그립도 더 털림
GRIP_GAIN = 0.25
# 땀: 구간마다 쌓이고(체감온도·강도에 비례) 조금씩 마른다. 땀이 많을수록 그립이 더 털린다.
SWEAT_GAIN = {"run": 0.03, "station": 0.05}
SWEAT_DECAY = 0.92
SWEAT_GRIP = 0.6
CHALK_SEC = 5.0          # 초크·수건 정리에 드는 시간
CHALK_SWEAT_KEEP = 0.4   # 초크 후 남는 땀 비율
# 컨디션: 50 = 평소. 0~100 → 시간 ±4%
COND_SWING = 0.04
# 온도: 체감온도 20℃ 초과 시 시간·스태미나 소모 증가, 8℃ 미만은 근육이 굳어 소폭 손해
HEAT_TIME, COLD_TIME, HEAT_DRAIN = 0.006, 0.003, 0.04


TRAIT_SWING = 0.25       # 스태미나/그립 능력치 50 기준 ±50 → 소모율 ±25%


def trait_mult(rating: float) -> float:
    return 1.0 - TRAIT_SWING * (rating - 50.0) / 50.0


def _drains(traits: dict[str, float] | None) -> tuple[float, float]:
    t = traits or {}
    return round(trait_mult(t.get("stamina", 50.0)), 3), round(trait_mult(t.get("grip", 50.0)), 3)


@dataclass
class Conditions:
    """레이스 전 조절 가능한 조건. 기본값(중립)에서는 개인 기준 기록과 동일."""
    condition: float = 50.0    # 컨디션 0~100 (50 = 평소)
    stamina: float = 100.0     # 시작 스태미나 % (수면·테이퍼링 등 준비도)
    grip: float = 100.0        # 시작 그립 % (손 상태·악력)
    temperature: float = 18.0  # ℃
    humidity: float = 50.0     # %

    @property
    def feels(self) -> float:
        t = self.temperature
        return t + 0.1 * (self.humidity - 50.0) if t > 20 else t

    @property
    def time_mult(self) -> float:
        f = self.feels
        cond = 1.0 - COND_SWING * (self.condition - 50.0) / 50.0
        return cond * (1.0 + HEAT_TIME * max(0.0, f - 20.0) + COLD_TIME * max(0.0, 8.0 - f))

    @property
    def stamina_drain_mult(self) -> float:
        return 1.0 + HEAT_DRAIN * max(0.0, self.feels - 20.0)

    @property
    def sweat_rate(self) -> float:
        return 1.0 + 0.10 * max(0.0, self.feels - 15.0)


@dataclass
class Strategy:
    """구간별 전략. run: 페이스, station: 스테이션 전략. 미지정 구간은 default."""
    default: str = "plan"
    run_pace: dict[str, str] = field(default_factory=dict)
    station: dict[str, str] = field(default_factory=dict)   # 스테이션명 → safe/steady/push
    chalk: set[str] = field(default_factory=set)            # 초크·수건을 쓸 그립 스테이션

    def pace_of(self, seg: str) -> str:
        return self.run_pace.get(seg, self.default)

    def multipliers(self, seg: str) -> tuple[float, float, float]:
        """(시간, 스태미나 소모, 그립 소모) 계수."""
        if KIND[seg] == "run":
            t, d = PACE[self.pace_of(seg)]
            return t, d, 1.0
        return STATION[self.station.get(seg, "steady")]


def _step(state: tuple[float, float, float], seg: str, mult: tuple[float, float, float],
          cond: Conditions, chalk: bool = False) -> tuple[float, float, float]:
    """(스태미나, 그립, 땀) 상태를 한 구간만큼 진행."""
    stamina, grip, sweat = state
    kind = KIND[seg]
    _, drain, grip_mult = mult
    stamina = max(0.0, stamina - STAMINA_DRAIN[kind] * drain * cond.stamina_drain_mult)
    sweat = min(1.0, sweat * SWEAT_DECAY + SWEAT_GAIN[kind] * cond.sweat_rate * drain)
    if seg in GRIP_SEGMENTS:
        if chalk:
            sweat *= CHALK_SWEAT_KEEP
        grip = max(0.0, grip - GRIP_DRAIN * grip_mult * (1 + GRIP_STAMINA_COUPLING * (1 - stamina))
                   * (1 + SWEAT_GRIP * sweat))
    return stamina, grip, sweat


@lru_cache(maxsize=256)
def _plan_trajectory(s_mult: float = 1.0, g_mult: float = 1.0) -> list[tuple[float, float]]:
    """중립 조건·계획 전략 시 구간 종료 시점의 (스태미나 결손, 그립 결손). 선수별 소모율(능력치)을 반영한다.

    계획 페이스 기대값 == 개인 기준 기록이 되도록, 피로는 '이 선수의 계획 궤적' 대비 초과분만 센다.
    """
    state, out, cond = (1.0, 1.0, 0.0), [], Conditions()
    for seg in NAMES:
        state = _step(state, seg, (1.0, s_mult, g_mult), cond)
        out.append((1 - state[0], 1 - state[1]))
    return out


def plan_reserve(traits: dict[str, float] | None = None) -> dict[str, float]:
    """계획 페이스 완주 시 남는 스태미나·그립(0~1) — 능력치 화면용."""
    s_end, g_end = _plan_trajectory(*_drains(traits))[-1]
    return {"stamina": 1 - s_end, "grip": 1 - g_end}


def simulate(profile: AthleteProfile, strategy: Strategy, rng: np.random.Generator,
             cond: Conditions | None = None) -> dict:
    cond = cond or Conditions()
    s_mult, g_mult = _drains(profile.traits)
    plan_traj = _plan_trajectory(s_mult, g_mult)
    state = (cond.stamina / 100, cond.grip / 100, 0.0)
    day = rng.lognormal(0.0, profile.day_sd)
    splits, total = {}, 0.0
    trace = []   # 구간 종료 시점의 (스태미나, 그립, 땀) — 게임 UI 게이지용
    for i, seg in enumerate(NAMES):
        t_mult, drain, grip_mult = strategy.multipliers(seg)
        mult = (t_mult, drain * s_mult, grip_mult * g_mult)
        chalk = seg in strategy.chalk and seg in GRIP_SEGMENTS
        state = _step(state, seg, mult, cond, chalk)
        stamina, grip, sweat = state
        plan_s, plan_g = plan_traj[i]
        fatigue = 1.0 + FATIGUE_GAIN * ((1 - stamina) ** 2 - plan_s**2)
        grip_pen = 1.0 + (GRIP_GAIN * ((1 - grip) ** 2 - plan_g**2) if seg in GRIP_SEGMENTS else 0.0)
        t = (profile.base[seg] * t_mult * fatigue * grip_pen * cond.time_mult * day
             * rng.lognormal(0.0, profile.cv[seg]) + (CHALK_SEC if chalk else 0.0))
        splits[seg] = t
        total += t
        trace.append({"stamina": stamina, "grip": grip, "sweat": sweat})
    # Roxzone/전환 시간: 페이스 전략과 무관하게 개인 기준 + 변동 (total_time = 16구간 합 + roxzone)
    rox = profile.base[OVERHEAD] * cond.time_mult * day * rng.lognormal(0.0, profile.cv[OVERHEAD])
    splits[OVERHEAD] = rox
    total += rox
    return {"splits": splits, "total": total, "stamina_end": state[0], "grip_end": state[1],
            "sweat_end": state[2], "trace": trace}

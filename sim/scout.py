"""능력치(스카우트 리포트): 개인 WOD·신체 정보 + 실제 경기 기록 → 개인 프로필.

1. 모집단(성별×종목) 통계를 DB에서 만든다: 17성분 로그 시간의 평균·공분산, 능력치별 분위표, 나이대별 오프셋. (데이터 기반)
2. 사전분포: 경기 기록이 있으면 그 프로필(보정 완료), 없으면 모집단 평균 + 나이 오프셋.
3. WOD 결과를 성분에 대한 관측값으로 바꿔 칼만 갱신으로 합친다. 측정하지 않은 성분은 모집단 공분산으로 추정된다.
   - 시간 기반 테스트(5K/2K 로우/1K 스키)는 HYROX 구간 시간으로 직접 환산한다 (환산 계수는 설계값).
   - 근력·짐내스틱 테스트는 참가자 기준 z점수로 바꿔 상관 ρ만큼 반영한다 (기준값·ρ는 설계값).
   → 설계값은 WOD와 실제 기록이 함께 있는 데이터가 쌓이면 회귀로 교체해야 한다.
4. 합쳐진 성분 벡터를 같은 집단 분위표에 대어 0~100 능력치로 만들고, 스태미나·그립 능력치는 엔진의 소모율(traits)로 쓴다.

경기 기록만 있고 WOD가 없으면 기준 기록은 build_profile 결과 그대로다 (모집단 쪽으로 당기지 않는다 — CLAUDE.md 참고).
"""
import json
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from . import data
from .config import CACHE_DIR, DB_PATH
from .engine import plan_reserve
from .model import cond_of, get_model
from .profile import AthleteProfile, build_profile
from .segments import ALL

POP_VERSION = 1
IDX = {n: i for i, n in enumerate(ALL)}
RUNS = [f"run{i}" for i in range(1, 9)]
MIN_GROUP = 150
MIN_BAND = 30

ATTRS = {   # 키 → 표시 이름
    "run": "러닝 파워", "engine": "유산소 엔진", "power": "전신 파워", "grip": "그립·근지구력",
    "burpee": "버피 지구력", "stamina": "스태미나", "transition": "전환·회복",
}


def _mean(v: np.ndarray, names: list[str]) -> float:
    return float(np.mean([v[IDX[n]] for n in names]))


def raw_metrics(v: np.ndarray) -> dict[str, float]:
    """로그 시간 벡터 → 능력치 원점수 (클수록 좋음)."""
    run, eng = _mean(v, RUNS), _mean(v, ["ski_erg", "rowing"])
    hold = _mean(v, RUNS[4:]) - _mean(v, RUNS[:4])          # 후반 러닝이 초반보다 얼마나 느려지는가
    return {
        "run": -run, "engine": -eng,
        "power": -_mean(v, ["sled_push", "sled_pull"]),
        "grip": -_mean(v, ["farmers", "sandbag", "wall_balls"]),
        "burpee": -float(v[IDX["burpee"]]),
        "transition": -float(v[IDX["roxzone"]]),
        "stamina": -0.4 * eng - 0.4 * run - 1.5 * hold,
        "overall": -float(np.log(np.exp(v).sum())),
    }


# ---- 모집단 -----------------------------------------------------------------

def _band_mid(cat: str | None) -> float | None:
    m = re.fullmatch(r"(\d\d)-(\d\d)", cat or "")
    return (int(m[1]) + int(m[2])) / 2 if m else None


def _build_population() -> dict:
    model = get_model()
    season = model.seasons[-1]
    persons = defaultdict(list)
    for r in data.load_singles():
        c, t = cond_of(r), (season, r["rt"], r["gender"])
        if c in model.beta and t in model.beta:
            persons[(r["person"], r["rt"], r["gender"])].append(
                (r["race"], np.log(r["x"]) - model.beta[c] + model.beta[t], r["cat"]))
    groups = defaultdict(list)
    for (_, rt, g), rs in persons.items():
        rs.sort(key=lambda x: x[0])
        groups[(g, rt)].append((np.mean([v for _, v, _ in rs], axis=0), _band_mid(rs[-1][2])))
    pop = {"version": POP_VERSION, "groups": {}, "age": {}}
    age_acc = defaultdict(lambda: defaultdict(list))
    for (g, rt), rows in groups.items():
        if len(rows) < MIN_GROUP:
            continue
        V = np.array([v for v, _ in rows])
        mu = V.mean(0)
        metrics = [raw_metrics(v) for v in V]
        pop["groups"][f"{g}|{rt}"] = {
            "n": len(rows), "mean": mu.tolist(), "cov": np.cov(V.T).tolist(),
            "q": {k: np.percentile([m[k] for m in metrics], np.linspace(0, 100, 101)).tolist() for k in metrics[0]},
        }
        for v, mid in rows:
            if mid is not None:
                age_acc[g][mid].append(v - mu)
    for g, bands in age_acc.items():
        pop["age"][g] = {str(mid): np.mean(vs, axis=0).tolist() for mid, vs in sorted(bands.items()) if len(vs) >= MIN_BAND}
    return pop


@lru_cache(maxsize=1)
def population() -> dict:
    path, model_path = CACHE_DIR / "population.json", CACHE_DIR / "model.json"
    newest = max(DB_PATH.stat().st_mtime, model_path.stat().st_mtime if model_path.exists() else 0)
    if path.exists() and path.stat().st_mtime >= newest:
        try:
            d = json.loads(path.read_text())
            if d.get("version") == POP_VERSION:
                return d
        except (ValueError, KeyError):
            pass
    d = _build_population()
    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(d))
    return d


def age_offset(pop: dict, gender: str, age: float | None) -> np.ndarray:
    bands = pop["age"].get(gender) or {}
    if age is None or len(bands) < 2:
        return np.zeros(len(ALL))
    mids = sorted(float(k) for k in bands)
    tab = np.array([bands[k] for k in sorted(bands, key=float)])
    return np.array([np.interp(age, mids, tab[:, j]) for j in range(len(ALL))])


# ---- WOD → 관측 -------------------------------------------------------------

def _w(names: list[str]) -> dict[str, float]:
    return {n: 1 / len(names) for n in names}


# 시간 기반: (이름, HYROX 성분 가중치, 환산식, 로그 sd). 환산 계수는 설계값.
TIMED = {
    "run5k": ("5K 러닝", _w(RUNS), lambda s: s / 5 * 1.08, 0.07),          # 1km 평균 페이스 → HYROX 러닝 구간(+8%)
    "row2k": ("2K 로우", {"rowing": 1.0}, lambda s: s / 2 * 1.06, 0.07),    # 2K 평균 1000m 페이스 → HYROX 로잉(+6%)
    "ski1k": ("1K 스키에르그", {"ski_erg": 1.0}, lambda s: s * 1.08, 0.07),
}
# 능력 기반: (이름, 성분 가중치, 성별 (평균, sd), 높을수록 좋음, 상관 ρ). 기준값은 HYROX 참가자 수준 가정(설계값).
SCORED = {
    "deadlift": ("데드리프트 1RM/체중", {"sled_push": 1 / 3, "sled_pull": 1 / 3, "farmers": 1 / 3}, {"M": (2.0, 0.4), "F": (1.6, 0.3)}, True, 0.45),
    "backsquat": ("백스쿼트 1RM/체중", {"sled_push": 0.4, "sandbag": 0.3, "wall_balls": 0.3}, {"M": (1.6, 0.35), "F": (1.2, 0.25)}, True, 0.40),
    "pullups": ("스트릭트 풀업", {"sled_pull": 0.4, "burpee": 0.3, "farmers": 0.3}, {"M": (10, 5), "F": (4, 3)}, True, 0.35),
    "dead_hang": ("데드행(초)", {"farmers": 0.5, "sandbag": 0.25, "sled_pull": 0.25}, {"M": (70, 30), "F": (45, 20)}, True, 0.40),
    "fran": ("Fran(초)", {"burpee": 0.4, "wall_balls": 0.6}, {"M": (330, 100), "F": (390, 120)}, False, 0.40),
    "cindy": ("Cindy(20분 라운드)", {"burpee": 0.3, "wall_balls": 0.3, "sandbag": 0.2, "run1": 0.2}, {"M": (16, 4), "F": (14, 4)}, True, 0.35),
    "wallball_2min": ("2분 월볼(개)", {"wall_balls": 1.0}, {"M": (55, 10), "F": (45, 10)}, True, 0.50),
}


def _vec(weights: dict[str, float]) -> np.ndarray:
    h = np.zeros(len(ALL))
    for n, w in weights.items():
        h[IDX[n]] = w
    return h


def observations(wod: dict, gender: str, weight_kg: float | None, pop_mean: np.ndarray, pop_cov: np.ndarray):
    """→ ([(h, y, sd)], [사용한 테스트 설명], [건너뛴 테스트 사유])."""
    obs, used, skipped = [], [], []
    for k, (label, weights, conv, sd) in TIMED.items():
        if wod.get(k):
            per = conv(wod[k])
            obs.append((_vec(weights), float(np.log(per)), sd))
            used.append(f"{label} → HYROX 환산 {per:.0f}s/구간")
    for k, (label, weights, norms, higher, rho) in SCORED.items():
        val = wod.get(k)
        if val is None:
            continue
        if k in ("deadlift", "backsquat"):
            if not weight_kg:
                skipped.append(f"{label}: 체중 입력 필요")
                continue
            val = val / weight_kg
        mu, sd = norms[gender]
        z = float(np.clip((val - mu) / sd * (1 if higher else -1), -3, 3))
        h = _vec(weights)
        sigma = float(np.sqrt(h @ pop_cov @ h))
        obs.append((h, float(h @ pop_mean - rho * sigma * z), sigma * float(np.sqrt(1 - rho**2))))
        used.append(f"{label} z={z:+.1f}")
    return obs, used, skipped


def kalman(m: np.ndarray, P: np.ndarray, obs) -> tuple[np.ndarray, np.ndarray]:
    if not obs:
        return m, P
    H = np.array([o[0] for o in obs])
    y = np.array([o[1] for o in obs])
    S = H @ P @ H.T + np.diag([o[2] ** 2 for o in obs])
    K = P @ H.T @ np.linalg.inv(S)
    return m + K @ (y - H @ m), P - K @ H @ P


def ratings(pop_g: dict, v: np.ndarray) -> dict[str, int]:
    raw, grid = raw_metrics(v), np.linspace(0, 100, 101)
    return {k: int(round(np.clip(np.interp(raw[k], pop_g["q"][k], grid), 1, 99))) for k in raw}


# ---- 진입점 -----------------------------------------------------------------

def build(name: str | None, nationality: str | None, bio: dict | None = None, wod: dict | None = None
          ) -> tuple[AthleteProfile, dict]:
    """name이 있으면 그 선수의 경기 기록이 사전분포, 없으면 모집단(성별·종목·나이)이 사전분포."""
    bio, wod = bio or {}, {k: v for k, v in (wod or {}).items() if v is not None}
    pop, model = population(), get_model()
    race = build_profile(name, nationality) if name else None
    gender = race.gender if race else bio.get("gender")
    rt = race.race_type if race else bio.get("race_type", "open")
    if gender not in ("M", "F"):
        raise ValueError("성별(M/F)이 필요합니다")
    g = pop["groups"].get(f"{gender}|{rt}")
    if g is None:
        raise ValueError(f"집단 데이터 부족: {gender}/{rt}")
    mu, cov = np.array(g["mean"]), np.array(g["cov"])
    cv = np.array(model.seg_cv)

    age = bio.get("age")
    age_off = np.zeros(len(ALL))
    if race:
        m0 = np.log([race.base[s] for s in ALL])
        sd0 = cv / np.sqrt(race.n_races)
        sd_pop = np.sqrt(np.diag(cov))
        P0 = np.outer(sd0, sd0) * (cov / np.outer(sd_pop, sd_pop))   # 나이 영향은 기록에 이미 들어 있다
    else:
        age_off = age_offset(pop, gender, age)
        m0, P0 = mu + age_off, cov

    obs, used, skipped = observations(wod, gender, bio.get("weight_kg"), mu, cov)
    m, P = kalman(m0, P0, obs)
    if race:
        base = dict(race.base) if not obs else {s: float(np.exp(v)) for s, v in zip(ALL, m)}
        day_sd, n_races, season = race.day_sd, race.n_races, race.season
        name_, nat, adjusted = race.name, race.nationality, race.adjusted
    else:
        base = {s: float(np.exp(v)) for s, v in zip(ALL, m)}
        day_sd, n_races, season = model.day_sd * 1.2, 0, model.seasons[-1]
        name_, nat, adjusted = bio.get("name") or "나", "", True
    cv_eff = np.sqrt(cv**2 + np.diag(P))   # 경기 간 변동 + 남은 수준 불확실성 (WOD 없으면 build_profile의 sqrt(1+1/n)과 동일)
    rating = ratings(g, m)
    traits = {"stamina": rating["stamina"], "grip": rating["grip"]}
    prof = AthleteProfile(name_, nat, gender, rt, season, n_races, base,
                          {s: float(v) for s, v in zip(ALL, cv_eff)}, float(day_sd), adjusted, traits=traits)
    rep = {
        "ratings": [{"key": k, "label": ATTRS.get(k, "종합"), "value": rating[k]} for k in [*ATTRS, "overall"]],
        "group": {"gender": gender, "race_type": rt, "n": g["n"]},
        "sources": {"races": n_races, "wod": used, "skipped": skipped, "age": None if race else age},
        "traits": traits, "reserve": plan_reserve(traits),
        "age_effect_pct": None if race or age is None else
            float(np.exp(mu + age_off).sum() / np.exp(mu).sum() * 100 - 100),
    }
    if race:
        rep["race_only_total"] = race.expected_total
        rep["race_only_ratings"] = ratings(g, np.log([race.base[s] for s in ALL]))
        rep["wod_shift_pct"] = float((prof.expected_total / race.expected_total - 1) * 100)
    return prof, rep

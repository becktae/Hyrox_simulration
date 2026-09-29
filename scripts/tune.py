"""백테스트 변형 비교 (메모리 상에서 빠르게). 대상: 2경기 이상 온전한 싱글 기록이 있는 선수의 마지막 경기.

사용: python -m scripts.tune
"""
from collections import defaultdict

import numpy as np

from sim import data
from sim.segments import COLUMNS, KIND, NAMES

RUN_IDX = [i for i, n in enumerate(NAMES) if KIND[n] == "run"]
ST_IDX = [i for i, n in enumerate(NAMES) if KIND[n] == "station"]


def load():
    return data.load_singles()


def main():
    rows = load()
    by_person = defaultdict(list)
    for r in rows:
        by_person[r["person"]].append(r)
    targets = []
    for p, rs in by_person.items():
        rs.sort(key=lambda r: r["race"])
        if len(rs) >= 2:
            targets.append((rs[-1], rs[:-1]))
    tid = {id(t) for t, _ in targets}
    pool = [r for r in rows if id(r) not in tid]            # 평가 대상 경기를 뺀 prior 풀
    print(f"rows={len(rows)} people>=2={len(targets)} prior pool={len(pool)}")

    groups = defaultdict(list)
    for r in pool:
        for key in [(r["season"], r["rt"], r["gender"], r["cat"]), (r["season"], r["rt"], r["gender"]),
                    (r["rt"], r["gender"], r["cat"]), (r["rt"], r["gender"])]:
            groups[key].append(r["x"])
    cache = {}

    def prior(t, use_season=True, use_cat=True):
        keys = []
        if use_season and use_cat: keys.append((t["season"], t["rt"], t["gender"], t["cat"]))
        if use_season: keys.append((t["season"], t["rt"], t["gender"]))
        if use_cat: keys.append((t["rt"], t["gender"], t["cat"]))
        keys.append((t["rt"], t["gender"]))
        for k in keys:
            if len(groups.get(k, [])) >= 30:
                if k not in cache:
                    cache[k] = np.mean(groups[k], axis=0)
                return cache[k]
        raise KeyError(keys)

    def evaluate(name, fn):
        e = np.array([fn(t, h).sum() - t["total"] for t, h in targets])
        print(f"{name:52s} MAE={np.abs(e).mean():5.0f}s bias={e.mean():+5.0f}s within120={(np.abs(e) <= 120).mean():4.0%}")

    def own_mean(h): return np.mean([r["x"] for r in h], axis=0)

    # ---- 시즌 정규화: 과거 경기를 목표 시즌 수준으로 환산 (같은 종목·성별 풀 평균 비율)
    season_mean = {}
    for r in pool:
        season_mean.setdefault((r["season"], r["rt"], r["gender"]), []).append(r["x"])
    season_mean = {k: np.mean(v, axis=0) for k, v in season_mean.items() if len(v) >= 30}

    def norm(r, t, per_seg=True):
        a, b = season_mean.get((t["season"], t["rt"], t["gender"])), season_mean.get((r["season"], r["rt"], r["gender"]))
        if a is None or b is None:
            return r["x"]
        return r["x"] * (a / b if per_seg else a.sum() / b.sum())

    def wmean(vs, lam):
        w = np.array([lam ** i for i in range(len(vs) - 1, -1, -1)], float)   # 최신 = 1
        return (np.array(vs) * w[:, None]).sum(0) / w.sum()

    for lam in (1.0, 0.6, 0.3):
        evaluate(f"H own mean, season-normalized, decay={lam}", lambda t, h, lam=lam: wmean([norm(r, t) for r in h], lam))
    evaluate("H2 total-only season norm, decay=0.6", lambda t, h: wmean([norm(r, t, False) for r in h], 0.6))
    evaluate("H3 last race only, season-normalized", lambda t, h: norm(h[-1], t))
    # 향상 추세: 정규화 후 (최신 - 평균)의 일부를 더함
    for a in (0.15, 0.3, 0.5):
        def f(t, h, a=a):
            vs = [norm(r, t) for r in h]
            m = wmean(vs, 1.0)
            return m + a * (vs[-1] - m) if len(vs) > 1 else m
        evaluate(f"I mean + {a}*(last-mean)", f)
    # 소량 prior (재출전 선수 풀) — 결측/희소 구간용
    for k in (0.05, 0.1):
        def f(t, h, k=k):
            n = len(h); w = n / (n + k)
            return w * wmean([norm(r, t) for r in h], 1.0) + (1 - w) * prior(t)
        evaluate(f"J season-norm mean + tiny prior K={k}", f)
    print()
    evaluate("A prior(cat only, no season)", lambda t, h: prior(t, False, True))
    evaluate("B prior(season+cat)", lambda t, h: prior(t))
    evaluate("C own mean only", lambda t, h: own_mean(h))
    for k in (0.25, 0.5, 1, 1.5, 3):
        def f(t, h, k=k):
            n = len(h); w = n / (n + k)
            return w * own_mean(h) + (1 - w) * prior(t)
        evaluate(f"D shrink K={k} (own+prior season/cat)", f)
    # E: 개인 = 카테고리 대비 '비율 팩터'(러닝/스테이션/록스존 별도) — 시즌 효과는 prior가 흡수
    for k in (0.5, 1, 2):
        def f(t, h, k=k):
            n = len(h); w = n / (n + k)
            base_t = prior(t)
            # 과거 경기 시점의 prior 대비 비율
            ratios = np.array([r["x"] / prior(r) for r in h]).mean(axis=0)
            groups_idx = [RUN_IDX, ST_IDX, [16]]
            fac = np.ones(17)
            for idx in groups_idx:
                fac[idx] = 1 + w * (ratios[idx].mean() - 1)
            return base_t * fac
        evaluate(f"E group-ratio K={k}", f)
    for k in (0.5, 1, 2):
        def f(t, h, k=k):
            n = len(h); w = n / (n + k)
            ratios = np.array([r["x"] / prior(r) for r in h]).mean(axis=0)
            return prior(t) * (1 + w * (ratios - 1))
        evaluate(f"F per-segment ratio K={k}", f)
    def g(t, h):
        ratios = np.array([r["x"].sum() / prior(r).sum() for r in h]).mean()
        return prior(t) * ratios
    evaluate("G single total ratio (K=0)", g)


if __name__ == "__main__":
    main()

"""개인 평균(방식 C)의 오차를 구간/집단별로 분해."""
from collections import defaultdict

import numpy as np

from scripts.tune import NAMES, load

rows = load()
bp = defaultdict(list)
for r in rows:
    bp[r["person"]].append(r)
T = []
for p, rs in bp.items():
    rs.sort(key=lambda r: r["race"])
    if len(rs) >= 2:
        T.append((rs[-1], rs[:-1]))
err = np.array([np.mean([r["x"] for r in h], axis=0) - t["x"] for t, h in T])   # +면 예측이 느림
print("n", len(T))
print("component bias (s):", {n: round(err[:, i].mean()) for i, n in enumerate(NAMES + ["overhead"])})
print("component MAE  (s):", {n: round(np.abs(err[:, i]).mean()) for i, n in enumerate(NAMES + ["overhead"])})
tot = err.sum(1)

def grp(label, keyf):
    d = defaultdict(list)
    for (t, h), e in zip(T, tot):
        d[keyf(t, h)].append(e)
    print(f"\n-- by {label}")
    for k in sorted(d):
        v = np.array(d[k])
        if len(v) >= 40:
            print(f"{str(k):40s} n={len(v):5d} bias={v.mean():+6.0f} MAE={np.abs(v).mean():5.0f} within120={(np.abs(v)<=120).mean():.0%}")

grp("target season", lambda t, h: t["season"])
grp("target rt", lambda t, h: t["rt"])
grp("hist last season -> target season", lambda t, h: f'{h[-1]["season"]}->{t["season"]}')
grp("rt change", lambda t, h: f'{h[-1]["rt"]}->{t["rt"]}')
grp("n history", lambda t, h: min(len(h), 4))
grp("history time quartile", lambda t, h: int(np.digitize(np.mean([r["total"] for r in h]), [3600, 4500, 5400])))
# 개인 내 변동의 하한: 이전 경기끼리의 차이(2경기 이상 히스토리)
d2 = [abs(h[-1]["total"] - np.mean([r["total"] for r in h[:-1]])) for t, h in T if len(h) >= 2]
print("\nwithin-person |last hist - mean earlier hist| (noise proxy) MAE:", round(np.mean(d2)), "n", len(d2))

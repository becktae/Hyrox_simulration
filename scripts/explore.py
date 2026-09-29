import sys
import numpy as np
from sim import data
from sim.segments import COLUMNS, NAMES

with data.connect() as con:
    rows = [dict(r) for r in con.execute(f"""SELECT a.name,a.nationality,a.gender,r.race_id,r.category,r.total_time,r.penalty_time,r.roxzone_time,
        {",".join('r.'+c for c in COLUMNS)} FROM athletes a JOIN results r ON r.athlete_id=a.id WHERE r.total_time>0""")]
rows = [r for r in rows if data.race_type(r["race_id"])]
print("singles rows", len(rows))
full = [r for r in rows if all((r[c] or 0) > 0 for c in COLUMNS)]
print("full-split singles", len(full))
d = np.array([r["total_time"] - sum(r[c] for c in COLUMNS) for r in full])
print("total - sum(splits): mean %.0f median %.0f p5 %.0f p95 %.0f" % (d.mean(), np.median(d), *np.percentile(d, [5, 95])))
print("penalty non-null:", sum(1 for r in full if r["penalty_time"]), "roxzone non-null:", sum(1 for r in full if r["roxzone_time"]))
# 같은 (이름,국적,race_id) 중복?
from collections import Counter
c = Counter((r["name"], r["nationality"], r["race_id"]) for r in rows)
print("dup (name,nat,race) :", sum(1 for v in c.values() if v > 1))
# 사람별 full 경기 수
p = Counter((r["name"], r["nationality"]) for r in full)
print("people>=2 full:", sum(1 for v in p.values() if v >= 2), ">=3:", sum(1 for v in p.values() if v >= 3))
# 동명이인 의심: 성별 혼재
g = {}
for r in rows: g.setdefault((r["name"], r["nationality"]), set()).add(r["gender"])
print("mixed gender keys:", sum(1 for v in g.values() if len(v - {None, ''}) > 1))
# race_id 정렬 vs season
print(sorted({r["race_id"].split("__")[0] for r in rows}))

print("\n== diff by season / roxzone availability")
by = {}
for r in full:
    s = r["race_id"].split("__")[0]
    diff = r["total_time"] - sum(r[c] for c in COLUMNS)
    by.setdefault((s, r["roxzone_time"] is not None), []).append((diff, r["roxzone_time"]))
for k in sorted(by):
    v = by[k]; dd = np.array([x[0] for x in v])
    extra = ""
    if k[1]:
        rz = np.array([x[1] for x in v]); extra = f" roxzone med {np.median(rz):.0f} corr(diff,rox)={np.corrcoef(dd, rz)[0,1]:.2f} med(diff-rox)={np.median(dd-rz):.0f}"
    print(k, f"n={len(v)} diff med {np.median(dd):.0f} p10 {np.percentile(dd,10):.0f} p90 {np.percentile(dd,90):.0f}{extra}")

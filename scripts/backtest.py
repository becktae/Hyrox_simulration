"""백테스트: 2경기 이상 온전한 싱글 기록이 있는 선수의 마지막 경기를 이전 기록만으로 예측.

- 조건 효과 모델은 사람 단위 5-fold로 적합 (평가 대상 사람은 적합에서 제외 → 누수 없음)
- 지표: P50 오차(MAE/편향/±120s 적중), 그리고 몬테카를로 P10~P90이 실제를 포함하는 비율(캘리브레이션, 목표 ≈80%)

사용: python -m scripts.backtest [--per-fold 300]
"""
import argparse
import zlib
from collections import defaultdict

import numpy as np

from sim import data
from sim.engine import Strategy
from sim.model import ConditionModel, cond_of
from sim.montecarlo import run
from sim.profile import build_profile


# 목표: P10~P90 포함률이 80% ± 5%p (MAE는 참고 지표 — 선수 본인의 경기 간 변동이 커서 ±120s는 비현실적)
COVERAGE_TARGET, COVERAGE_TOL = 0.80, 0.05


def main(per_fold: int, n_sims: int):
    rows = data.load_singles()
    by = defaultdict(list)
    for r in rows:
        by[r["person"]].append(r)
    people = {p: sorted(v, key=lambda r: r["race"]) for p, v in by.items() if len(v) >= 2}
    fold = {p: zlib.crc32(repr(p).encode()) % 5 for p in people}
    out = []
    for k in range(5):
        model = ConditionModel.fit([r for p, v in people.items() if fold[p] != k for r in v])
        mine = [p for p in people if fold[p] == k][:per_fold]
        for p in mine:
            t, h = people[p][-1], people[p][:-1]
            if t["season"] not in model.seasons:
                continue
            prof = build_profile(p[0], p[1], season=t["season"], race_type=t["rt"], model=model, hist=h)
            m = run(prof, Strategy(), n=n_sims, seed=0)
            out.append((m["p50"] - t["total"], m["p10"] <= t["total"] <= m["p90"], cond_of(t) == cond_of(h[-1]), t["season"]))
    def show(label, sel):
        e = np.array([x[0] for x in sel])
        cov = np.mean([x[1] for x in sel])
        ok = "PASS" if abs(cov - COVERAGE_TARGET) <= COVERAGE_TOL else "FAIL"
        print(f"{label:24s} n={len(e):5d} MAE={np.abs(e).mean():4.0f}s bias={e.mean():+5.0f}s "
              f"within120={(np.abs(e) <= 120).mean():4.0%}  P10-P90 coverage={cov:4.0%} [{ok}]")
    show("ALL", out)
    show("same condition", [x for x in out if x[2]])
    show("condition changed", [x for x in out if not x[2]])
    show("target season-8", [x for x in out if x[3] == "season-8"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-fold", type=int, default=300)
    ap.add_argument("--sims", type=int, default=300)
    a = ap.parse_args()
    main(a.per_fold, a.sims)

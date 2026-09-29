"""사람 고정효과 + 조건(시즌·종목·성별) 효과 모델. 사람 단위 5-fold로 누수 없이 평가.

log x[p,c,seg] = alpha[p,seg] + beta[c,seg]  →  예측 = alpha_hat(과거 경기, 조건 보정) + beta[target 조건]
"""
from collections import defaultdict
import zlib

import numpy as np

from scripts.tune import NAMES, load


def cond(r):
    return (r["season"], r["rt"], r["gender"])


def fit_beta(train_rows, iters=15, min_rows=30):
    """train_rows: 2경기 이상 가진 사람의 모든 경기. 반환 {cond: beta(17)}"""
    by_p = defaultdict(list)
    for r in train_rows:
        by_p[r["person"]].append(r)
    by_p = {p: v for p, v in by_p.items() if len(v) >= 2}
    conds = sorted({cond(r) for v in by_p.values() for r in v})
    cid = {c: i for i, c in enumerate(conds)}
    beta = np.zeros((len(conds), 17))
    cnt = np.zeros(len(conds))
    P = [(np.array([np.log(r["x"]) for r in v]), np.array([cid[cond(r)] for r in v])) for v in by_p.values()]
    for _, ci in P:
        for c in ci:
            cnt[c] += 1
    for _ in range(iters):
        num = np.zeros_like(beta)
        for Y, ci in P:
            a = (Y - beta[ci]).mean(0)
            np.add.at(num, ci, Y - a)
        new = num / np.maximum(cnt, 1)[:, None]
        beta = new - new[cnt.argmax()]        # 최다 조건을 기준(0)으로 고정
    return {c: beta[i] for c, i in cid.items() if cnt[i] >= min_rows}


def predict(t, h, beta, lam=1.0):
    bt = beta.get(cond(t))
    ys, ws = [], []
    for i, r in enumerate(h):
        br = beta.get(cond(r))
        if bt is None or br is None:
            br, bt_use = 0.0, 0.0      # 모르는 조건은 보정 없이
        else:
            bt_use = bt
        ys.append(np.log(r["x"]) - br + bt_use)
        ws.append(lam ** (len(h) - 1 - i))
    ws = np.array(ws)
    return np.exp((np.array(ys) * ws[:, None]).sum(0) / ws.sum())


def main():
    rows = load()
    bp = defaultdict(list)
    for r in rows:
        bp[r["person"]].append(r)
    people = {p: sorted(v, key=lambda r: r["race"]) for p, v in bp.items() if len(v) >= 2}
    fold = {p: zlib.crc32(repr(p).encode()) % 5 for p in people}
    res = defaultdict(list)
    for k in range(5):
        train = [r for p, v in people.items() if fold[p] != k for r in v]
        beta = fit_beta(train)
        for p, v in people.items():
            if fold[p] != k:
                continue
            t, h = v[-1], v[:-1]
            plain = np.mean([r["x"] for r in h], axis=0).sum()
            for lam in (1.0, 0.5):
                res[f"FE lam={lam}"].append((predict(t, h, beta, lam).sum() - t["total"], t, h))
            res["C plain own mean"].append((plain - t["total"], t, h))
    def show(name, items, label=""):
        e = np.array([x[0] for x in items])
        print(f"{name:22s}{label:28s} n={len(e):5d} MAE={np.abs(e).mean():5.0f}s bias={e.mean():+5.0f}s within120={(np.abs(e)<=120).mean():4.0%}")
    for name, items in res.items():
        show(name, items, "ALL")
    for name in ("C plain own mean", "FE lam=1.0"):
        items = res[name]
        show(name, [x for x in items if x[1]["season"] == "season-8"], "target=S8")
        show(name, [x for x in items if cond(x[1]) == cond(x[2][-1])], "same cond as last")
        show(name, [x for x in items if x[2][-1]["rt"] != x[1]["rt"]], "rt changed")
        show(name, [x for x in items if x[2][-1]["season"] != x[1]["season"]], "season changed")
    print("\nbeta S8 vs S7 (open, M) total-time ratio:", end=" ")
    beta = fit_beta([r for v in people.values() for r in v])
    for a, b in [(("season-7", "open", "M"), ("season-8", "open", "M")), (("season-8", "pro", "M"), ("season-8", "open", "M"))]:
        if a in beta and b in beta:
            print(f"{a[0]}/{a[1]}->{b[0]}/{b[1]}: {np.exp(beta[b]-beta[a]).round(3)[[0,3,5,15,16]]} (run1,sled_push,sled_pull,wall_balls,overhead)", end=" | ")
    print()


if __name__ == "__main__":
    main()

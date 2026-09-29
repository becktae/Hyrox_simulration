from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sim import data, montecarlo
from sim.engine import Strategy, simulate
from sim.profile import build_profile

import numpy as np

app = FastAPI(title="Hyrox Simulation")
FRONT = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=FRONT), name="static")


class SimRequest(BaseModel):
    name: str
    nationality: str
    default_pace: str = "plan"
    run_pace: dict[str, str] = {}
    n: int = 1000
    target: float | None = None
    seed: int | None = None


def _profile(name: str, nationality: str):
    try:
        return build_profile(name, nationality)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")


@app.get("/api/athletes")
def athletes(q: str, limit: int = 20):
    if len(q.strip()) < 2:
        raise HTTPException(400, "검색어는 2자 이상")
    return data.search_athletes(q.strip(), limit)


@app.get("/api/profile")
def profile(name: str, nationality: str):
    p = _profile(name, nationality)
    return {**p.__dict__, "expected_total": p.expected_total}


@app.post("/api/simulate")
def simulate_once(req: SimRequest):
    p = _profile(req.name, req.nationality)
    rng = np.random.default_rng(req.seed)
    return simulate(p, Strategy(req.default_pace, req.run_pace), rng)


@app.post("/api/montecarlo")
def monte(req: SimRequest):
    p = _profile(req.name, req.nationality)
    return montecarlo.run(p, Strategy(req.default_pace, req.run_pace),
                          n=min(req.n, 5000), seed=req.seed, target=req.target)

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sim import data, montecarlo, scout, store
from sim.engine import PACE, STATION, Conditions, Strategy, simulate

import numpy as np

app = FastAPI(title="Hyrox Simulation")
FRONT = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=FRONT), name="static")


FIELD_LABEL = {"run5k": "5K 러닝", "row2k": "2K 로우", "ski1k": "1K 스키에르그", "deadlift": "데드리프트 1RM", "backsquat": "백스쿼트 1RM",
               "pullups": "스트릭트 풀업", "dead_hang": "데드행", "fran": "Fran", "cindy": "Cindy", "wallball_2min": "2분 월볼",
               "age": "나이", "weight_kg": "체중", "gender": "성별", "race_type": "종목", "label": "선수 이름", "memo": "메모",
               "temperature": "온도", "humidity": "습도", "condition": "컨디션", "stamina": "시작 스태미나", "grip": "시작 그립"}
TIME_FIELDS = {"run5k", "row2k", "ski1k", "fran"}


def _show(field: str, v) -> str:
    if field in TIME_FIELDS and isinstance(v, (int, float)):
        return f"{int(v // 60)}:{int(round(v % 60)):02d}"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


@app.exception_handler(RequestValidationError)
async def validation_error(_, exc: RequestValidationError):
    """422를 '어느 항목이 왜 틀렸는지' 한 줄 한국어 문장으로 돌려준다 (화면에 그대로 표시)."""
    msgs = []
    for e in exc.errors():
        f, ctx, t = str(e["loc"][-1]), e.get("ctx", {}), e["type"]
        if t == "greater_than_equal":
            why = f"{_show(f, ctx['ge'])} 이상이어야 합니다"
        elif t == "less_than_equal":
            why = f"{_show(f, ctx['le'])} 이하여야 합니다"
        elif t in ("float_parsing", "float_type", "int_parsing"):
            why = "숫자로 입력하세요"
        elif t == "string_pattern_mismatch":
            why = "올바른 값을 선택하세요"
        elif t in ("string_too_short", "string_too_long"):
            why = "길이를 확인하세요"
        else:
            why = e["msg"]
        got = e.get("input")
        msgs.append(f"{FIELD_LABEL.get(f, f)}: {why}" + (f" (입력값 {_show(f, got)})" if isinstance(got, (int, float)) else ""))
    return JSONResponse({"detail": " / ".join(msgs)}, status_code=422)


class CondModel(BaseModel):
    condition: float = Field(50, ge=0, le=100)
    stamina: float = Field(100, ge=30, le=100)
    grip: float = Field(100, ge=30, le=100)
    temperature: float = Field(18, ge=-10, le=45)
    humidity: float = Field(50, ge=0, le=100)


class BioModel(BaseModel):
    """신체·종목 정보. 선수를 고르지 않고 내 WOD만으로 시작할 때는 gender/race_type이 필수."""
    gender: str | None = Field(None, pattern="^[MF]$")
    race_type: str = Field("open", pattern="^(open|pro)$")
    age: float | None = Field(None, ge=14, le=80)
    weight_kg: float | None = Field(None, ge=35, le=180)
    name: str | None = None


class WodModel(BaseModel):
    run5k: float | None = Field(None, ge=840, le=2700)        # 초
    row2k: float | None = Field(None, ge=360, le=780)         # 초
    ski1k: float | None = Field(None, ge=150, le=400)         # 초
    deadlift: float | None = Field(None, ge=20, le=400)       # kg 1RM
    backsquat: float | None = Field(None, ge=20, le=350)      # kg 1RM
    pullups: float | None = Field(None, ge=0, le=60)
    dead_hang: float | None = Field(None, ge=0, le=300)       # 초
    fran: float | None = Field(None, ge=100, le=900)          # 초
    cindy: float | None = Field(None, ge=0, le=40)            # 20분 라운드
    wallball_2min: float | None = Field(None, ge=0, le=120)


class SimRequest(BaseModel):
    name: str = ""            # 비우면 선수 없이 bio + wod 로 만든 가상 선수
    nationality: str = ""
    bio: BioModel = BioModel()
    wod: WodModel = WodModel()
    default_pace: str = "plan"
    run_pace: dict[str, str] = {}
    station_pace: dict[str, str] = {}
    chalk: list[str] = []
    conditions: CondModel = CondModel()
    n: int = 1000
    target: float | None = None
    seed: int | None = None


def _strategy(req: SimRequest) -> Strategy:
    if (set(req.run_pace.values()) | {req.default_pace}) - set(PACE) or set(req.station_pace.values()) - set(STATION):
        raise HTTPException(400, "알 수 없는 전략 값")
    return Strategy(req.default_pace, req.run_pace, req.station_pace, set(req.chalk))


def _scout(req: SimRequest):
    try:
        return scout.build(req.name or None, req.nationality or None, req.bio.model_dump(),
                           req.wod.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(404 if req.name else 400, str(e))


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
    p, _ = _scout(SimRequest(name=name, nationality=nationality))
    return {**p.__dict__, "expected_total": p.expected_total}


@app.post("/api/scout")
def scout_report(req: SimRequest):
    """능력치 리포트 + 합쳐진 프로필(기준 기록). 선수 선택 여부와 WOD 입력 여부에 상관없이 호출 가능."""
    p, rep = _scout(req)
    return {**rep, "profile": {**p.__dict__, "expected_total": p.expected_total}}


class SaveRequest(BaseModel):
    label: str = Field(min_length=1, max_length=40)
    memo: str = Field("", max_length=200)
    id: str | None = None      # 있으면 그 선수를 덮어쓴다
    name: str = ""
    nationality: str = ""
    bio: BioModel = BioModel()
    wod: WodModel = WodModel()


class PatchRequest(BaseModel):
    label: str | None = Field(None, min_length=1, max_length=40)
    memo: str | None = Field(None, max_length=200)


def _snapshot(name: str, nationality: str, bio: dict, wod: dict) -> dict:
    """현재 모델로 계산한 능력치 요약 (등록 시점·재계산 시점의 값)."""
    p, rep = _scout(SimRequest(name=name, nationality=nationality, bio=bio, wod=wod))
    ratings = {r["key"]: {"value": r["value"], "grade": r["grade"], "label": r["label"]} for r in rep["ratings"]}
    return {"overall": ratings["overall"]["value"], "grade": ratings["overall"]["grade"],
            "expected_total": p.expected_total, "gender": p.gender, "race_type": p.race_type, "ratings": ratings}


@app.get("/api/profiles")
def profiles_list():
    return store.list_all()


@app.post("/api/profiles")
def profiles_save(req: SaveRequest):
    """선수 등록/덮어쓰기: 입력(선수·신체·WOD)과 능력치 요약을 함께 저장한다."""
    bio, wod = req.bio.model_dump(exclude_none=True), req.wod.model_dump(exclude_none=True)
    return store.save({"label": req.label.strip(), "memo": req.memo.strip(), "name": req.name,
                       "nationality": req.nationality, "bio": bio, "wod": wod,
                       "snapshot": _snapshot(req.name, req.nationality, bio, wod)}, req.id)


@app.patch("/api/profiles/{profile_id}")
def profiles_patch(profile_id: str, req: PatchRequest):
    """이름·메모만 수정 (능력치 재계산 없음)."""
    fields = {k: v.strip() for k, v in req.model_dump(exclude_none=True).items()}
    rec = store.update(profile_id, fields)
    if rec is None:
        raise HTTPException(404, "등록된 선수가 없습니다")
    return rec


@app.post("/api/profiles/refresh")
def profiles_refresh():
    """모델이 바뀐 뒤 등록된 선수들의 능력치·기준 기록을 다시 계산한다."""
    out = []
    for p in store.list_all():
        try:
            snap = _snapshot(p["name"], p["nationality"], p["bio"], p["wod"])
        except HTTPException:
            continue
        out.append(store.update(p["id"], {"snapshot": snap}))
    return {"refreshed": len(out)}


@app.get("/api/profiles/trash")
def profiles_trash():
    return store.list_deleted()


@app.delete("/api/profiles/{profile_id}")
def profiles_delete(profile_id: str):
    """삭제 = 휴지통으로 이동. 기록은 지워지지 않고 복구할 수 있다."""
    if not store.delete(profile_id):
        raise HTTPException(404, "등록된 선수가 없습니다")
    return {"ok": True, "trash": True}


@app.post("/api/profiles/{profile_id}/restore")
def profiles_restore(profile_id: str):
    rec = store.restore(profile_id)
    if rec is None:
        raise HTTPException(404, "휴지통에 그 선수가 없습니다")
    return rec


class RevertRequest(BaseModel):
    index: int


@app.post("/api/profiles/{profile_id}/revert")
def profiles_revert(profile_id: str, req: RevertRequest):
    """이전 버전(history[index])으로 되돌린다. 되돌리기 직전 상태도 history에 남는다."""
    rec = store.revert(profile_id, req.index)
    if rec is None:
        raise HTTPException(404, "되돌릴 버전이 없습니다")
    return rec


@app.post("/api/simulate")
def simulate_once(req: SimRequest):
    p, _ = _scout(req)
    rng = np.random.default_rng(req.seed)
    return simulate(p, _strategy(req), rng, Conditions(**req.conditions.model_dump()))


@app.post("/api/montecarlo")
def monte(req: SimRequest):
    p, _ = _scout(req)
    return montecarlo.run(p, _strategy(req), n=min(req.n, 5000), seed=req.seed, target=req.target,
                          cond=Conditions(**req.conditions.model_dump()))

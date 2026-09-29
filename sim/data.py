"""hyrox.db 읽기 전용 접근. 수집 에이전트 DB는 절대 쓰지 않는다."""
import sqlite3
from dataclasses import dataclass

import numpy as np

from .config import DB_PATH, OVERHEAD_RANGE
from .segments import COLUMNS, SINGLES


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def race_type(race_id: str) -> str | None:
    """'season-8__HPRO_LR3..' -> 'pro'. 싱글이 아니면 None."""
    try:
        code = race_id.split("__", 1)[1].split("_", 1)[0]
    except IndexError:
        return None
    return SINGLES.get(code)


@dataclass(frozen=True)
class AthleteKey:
    name: str
    nationality: str


def search_athletes(q: str, limit: int = 20) -> list[dict]:
    """이름 부분일치 검색. athlete_id가 경기마다 달라 (이름, 국적)으로 묶는다."""
    sql = """
        SELECT a.name, a.nationality, MAX(a.gender) gender,
               COUNT(DISTINCT r.race_id) races, MIN(r.total_time) best
        FROM athletes a JOIN results r ON r.athlete_id = a.id
        WHERE a.name LIKE ? AND r.total_time > 0
        GROUP BY a.name, a.nationality
        ORDER BY races DESC, a.name LIMIT ?
    """
    with connect() as con:
        return [dict(r) for r in con.execute(sql, (f"%{q}%", limit))]


def athlete_history(name: str, nationality: str) -> list[dict]:
    """싱글 경기 기록만, 같은 (race_id) 중복 제거, 시간순(race_id의 season 기준)."""
    sql = f"""
        SELECT r.race_id, r.category, a.gender, r.total_time, r.penalty_time,
               {", ".join("r." + c for c in COLUMNS)}
        FROM athletes a JOIN results r ON r.athlete_id = a.id
        WHERE a.name = ? AND a.nationality = ? AND r.total_time > 0
    """
    seen, out = set(), []
    with connect() as con:
        for row in con.execute(sql, (name, nationality)):
            d = dict(row)
            rt = race_type(d["race_id"])
            key = (d["race_id"])
            if rt is None or key in seen:
                continue
            seen.add(key)
            d["race_type"] = rt
            out.append(d)
    out.sort(key=lambda d: d["race_id"])
    return out


def load_singles(name: str | None = None, nationality: str | None = None) -> list[dict]:
    """모델용 싱글 경기 행. 16구간이 모두 있고 total-합(roxzone)이 정상 범위인 행만.

    x = 17성분(16구간 + roxzone) 초 단위. (이름, 국적, race_id) 중복과, 다른 race_id에 똑같이 저장된
    동일 기록(스플릿 전부 일치)은 제거 — 남겨두면 백테스트에 누수가 생긴다.
    """
    sql = f"""SELECT a.name, a.nationality, a.gender, r.race_id, r.category, r.total_time,
              {", ".join("r." + c for c in COLUMNS)}
              FROM athletes a JOIN results r ON r.athlete_id = a.id WHERE r.total_time > 0"""
    params: tuple = ()
    if name is not None:
        sql += " AND a.name = ? AND a.nationality = ?"
        params = (name, nationality)
    out, seen, same_race = [], set(), set()
    lo, hi = OVERHEAD_RANGE
    with connect() as con:
        for r in con.execute(sql, params):
            rt = race_type(r["race_id"])
            if not rt or not r["gender"] or not all((r[c] or 0) > 0 for c in COLUMNS):
                continue
            key = (r["name"], r["nationality"], r["race_id"])
            if key in seen:
                continue
            v = np.array([r[c] for c in COLUMNS], float)
            over = r["total_time"] - v.sum()
            if not lo <= over <= hi:
                continue
            # 같은 경기가 다른 race_id(예: ..._OVERALL 집계 목록)로 중복 저장된 경우 제거
            fp = (key[0], key[1], tuple(np.round(v).astype(int)))
            if fp in same_race:
                continue
            seen.add(key)
            same_race.add(fp)
            out.append(dict(person=(r["name"], r["nationality"]), race=r["race_id"],
                            season=r["race_id"].split("__")[0], rt=rt, gender=r["gender"],
                            cat=r["category"], x=np.append(v, over), total=r["total_time"]))
    return _drop_near_duplicates(out)


NEAR_DUP_SEC = 3


def _drop_near_duplicates(rows: list[dict]) -> list[dict]:
    """같은 사람·종목에서 total_time 차이가 NEAR_DUP_SEC 이하인 기록은 같은 경기의 중복 저장으로 보고 하나만 남긴다.

    같은 이벤트가 heat/목록별 race_id로 따로 저장되고 스플릿이 반올림 차이로 살짝 달라 지문 비교를 빠져나간다.
    남겨두면 '이전 경기'가 평가 대상 경기의 복사본이 되어 백테스트가 크게 낙관적으로 나온다.
    """
    by = {}
    for r in sorted(rows, key=lambda r: (r["total"], r["race"])):
        by.setdefault((r["person"], r["rt"]), []).append(r)
    drop = set()
    for rs in by.values():
        last = None
        for r in rs:
            if last is not None and r["total"] - last["total"] <= NEAR_DUP_SEC:
                drop.add(id(r))
            else:
                last = r
    return [r for r in rows if id(r) not in drop]

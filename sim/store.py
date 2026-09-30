"""저장된 개인 프로필(JSON 파일). 서버에 보관하므로 폰·PC 어디서 접속해도 같은 목록이 보인다.

hyrox.db(수집 DB)와 별개 파일이며, 이 모듈만 쓰기를 한다."""
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

PATH = Path(os.environ.get("HYROX_PROFILES", Path(__file__).resolve().parents[1] / "saved" / "profiles.json"))
_LOCK = threading.Lock()

# 능력치 점수(1~99 백분위) → 등급. 점수 = 100 - 상위%
GRADES = [(95, "S"), (85, "A"), (70, "B"), (50, "C"), (30, "D"), (10, "E"), (0, "F")]


def grade_of(score: float) -> str:
    return next(g for lo, g in GRADES if score >= lo)


def _read() -> list[dict]:
    try:
        return json.loads(PATH.read_text())
    except (FileNotFoundError, ValueError):
        return []


def _write(items: list[dict]) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    tmp.replace(PATH)


def list_all() -> list[dict]:
    with _LOCK:
        return sorted(_read(), key=lambda p: p["updated"], reverse=True)


def save(item: dict, profile_id: str | None = None) -> dict:
    """profile_id가 있고 존재하면 덮어쓰기, 아니면 새로 만든다."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _LOCK:
        items = _read()
        old = next((p for p in items if p["id"] == profile_id), None) if profile_id else None
        rec = {**item, "id": old["id"] if old else uuid.uuid4().hex[:12],
               "created": old["created"] if old else now, "updated": now}
        items = [p for p in items if p["id"] != rec["id"]] + [rec]
        _write(items)
    return rec


def delete(profile_id: str) -> bool:
    with _LOCK:
        items = _read()
        keep = [p for p in items if p["id"] != profile_id]
        if len(keep) == len(items):
            return False
        _write(keep)
    return True

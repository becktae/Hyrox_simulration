"""등록된 선수(프로필) 저장소 — JSON 파일. 서버에 보관하므로 폰·PC 어디서 접속해도 같은 목록이 보인다.

한번 저장한 기록은 사라지지 않게 만든다:
- 삭제는 휴지통 이동(deleted 시각 기록)이며, 영구 삭제 기능은 없다. 복구 가능.
- 덮어쓰기·이름/메모 수정 때마다 이전 버전을 history에 남긴다 (최근 HISTORY_MAX개). 이전 버전으로 되돌릴 수 있다.
- 파일을 쓰기 전에 직전 파일을 backups/에 복사한다 (최근 BACKUP_MAX개).
- 파일이 깨졌으면 그 파일은 corrupt-*.json으로 보존하고 가장 최근 정상 백업에서 복원한다 (빈 목록으로 덮어쓰지 않는다).

hyrox.db(수집 DB)와 별개 파일이며, 이 모듈만 쓰기를 한다."""
import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

PATH = Path(os.environ.get("HYROX_PROFILES", Path(__file__).resolve().parents[1] / "saved" / "profiles.json"))
BACKUP_MAX, HISTORY_MAX = 300, 30
_LOCK = threading.Lock()
VERSIONED = ("label", "memo", "bio", "wod", "name", "nationality", "snapshot")   # history에 남기는 필드

# 능력치 점수(1~99 백분위) → 등급. 점수 = 100 - 상위%
GRADES = [(95, "S"), (85, "A"), (70, "B"), (50, "C"), (30, "D"), (10, "E"), (0, "F")]


def grade_of(score: float) -> str:
    return next(g for lo, g in GRADES if score >= lo)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def _backups() -> list[Path]:
    return sorted((PATH.parent / "backups").glob("profiles-*.json"))


def _load(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    if not isinstance(d, list):
        raise ValueError("not a list")
    return d


def _read() -> list[dict]:
    if not PATH.exists():
        return []
    try:
        return _load(PATH)
    except ValueError:
        PATH.rename(PATH.with_name(f"corrupt-{_stamp()}.json"))      # 증거 보존
        for b in reversed(_backups()):
            try:
                items = _load(b)
            except ValueError:
                continue
            _write(items, backup=False)
            return items
        return []


def _write(items: list[dict], backup: bool = True) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    if backup and PATH.exists() and PATH.stat().st_size > 2:
        bdir = PATH.parent / "backups"
        bdir.mkdir(exist_ok=True)
        shutil.copy2(PATH, bdir / f"profiles-{_stamp()}.json")
        for old in _backups()[:-BACKUP_MAX]:
            old.unlink()
    tmp = PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    tmp.replace(PATH)


def _push_history(rec: dict) -> None:
    rec["history"] = (rec.get("history", []) + [{**{k: rec[k] for k in VERSIONED if k in rec}, "saved": rec["updated"]}])[-HISTORY_MAX:]


def list_all(include_deleted: bool = False) -> list[dict]:
    with _LOCK:
        items = _read()
    items = [p for p in items if include_deleted or not p.get("deleted")]
    return sorted(items, key=lambda p: p["updated"], reverse=True)


def list_deleted() -> list[dict]:
    with _LOCK:
        return sorted((p for p in _read() if p.get("deleted")), key=lambda p: p["deleted"], reverse=True)


def save(item: dict, profile_id: str | None = None) -> dict:
    """profile_id가 있고 존재하면 덮어쓰기(이전 버전은 history에 보관), 아니면 새로 만든다."""
    now = _now()
    with _LOCK:
        items = _read()
        old = next((p for p in items if p["id"] == profile_id), None) if profile_id else None
        rec = {**item, "id": old["id"] if old else uuid.uuid4().hex[:12],
               "created": old["created"] if old else now, "updated": now, "history": []}
        if old:
            _push_history(old)                 # 덮어쓰기 직전 상태를 남긴다
            rec["history"] = old["history"]
        items = [p for p in items if p["id"] != rec["id"]] + [rec]
        _write(items)
    return rec


def update(profile_id: str, fields: dict) -> dict | None:
    """일부 필드만 바꾼다 (updated 시각 유지). 이름·메모가 바뀌면 이전 값을 history에 남긴다. 없으면 None."""
    with _LOCK:
        items = _read()
        rec = next((p for p in items if p["id"] == profile_id), None)
        if rec is None:
            return None
        if any(k in fields and fields[k] != rec.get(k) for k in ("label", "memo")):
            _push_history(rec)
        rec.update(fields)
        _write(items)
    return rec


def delete(profile_id: str) -> bool:
    """영구 삭제가 아니라 휴지통으로 옮긴다. 이미 휴지통이거나 없으면 False."""
    with _LOCK:
        items = _read()
        rec = next((p for p in items if p["id"] == profile_id and not p.get("deleted")), None)
        if rec is None:
            return False
        rec["deleted"] = _now()
        _write(items)
    return True


def restore(profile_id: str) -> dict | None:
    with _LOCK:
        items = _read()
        rec = next((p for p in items if p["id"] == profile_id and p.get("deleted")), None)
        if rec is None:
            return None
        rec.pop("deleted")
        _write(items)
    return rec


def revert(profile_id: str, index: int) -> dict | None:
    """history[index] 버전으로 되돌린다 (되돌리기 직전 상태도 history에 남는다)."""
    with _LOCK:
        items = _read()
        rec = next((p for p in items if p["id"] == profile_id), None)
        if rec is None or not -len(rec.get("history", [])) <= index < len(rec.get("history", [])):
            return None
        version = rec["history"][index]
        _push_history(rec)
        rec.update({k: version[k] for k in VERSIONED if k in version})
        rec["updated"] = _now()
        _write(items)
    return rec

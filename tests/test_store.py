import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HYROX_PROFILES", str(tmp_path / "p.json"))
    from sim import store as s
    return importlib.reload(s)


def test_grade_bands():
    from sim.store import grade_of
    assert [grade_of(x) for x in (99, 95, 90, 85, 75, 70, 60, 50, 40, 30, 20, 10, 5, 1)] == \
        list("SSAABBCCDDEEFF")


def test_save_overwrite_keeps_history(store):
    a = store.save({"label": "a", "memo": "1"})
    a2 = store.save({"label": "a2", "memo": "2"}, a["id"])         # 덮어쓰기: id·created 유지, 이전 버전 보관
    assert a2["id"] == a["id"] and a2["created"] == a["created"]
    assert [h["label"] for h in a2["history"]] == ["a"] and a2["history"][0]["memo"] == "1"
    assert len(store.list_all()) == 1


def test_delete_is_soft_and_restorable(store):
    a = store.save({"label": "a"})
    b = store.save({"label": "b"})
    assert store.delete(b["id"]) and not store.delete(b["id"])
    assert [p["label"] for p in store.list_all()] == ["a"]
    assert [p["label"] for p in store.list_deleted()] == ["b"]
    assert store.restore(b["id"])["label"] == "b" and not store.list_deleted()
    assert {p["label"] for p in store.list_all()} == {"a", "b"}
    assert store.restore("nope") is None


def test_backup_before_every_write_and_corrupt_recovery(store):
    a = store.save({"label": "a"})
    store.save({"label": "b"})
    assert len(store._backups()) >= 1
    store.PATH.write_text("{ broken json")                        # 파일 손상
    labels = {p["label"] for p in store.list_all()}
    assert labels >= {"a"}                                        # 빈 목록으로 덮어쓰지 않고 백업에서 복원
    assert list(store.PATH.parent.glob("corrupt-*.json"))         # 손상 파일은 보존


def test_revert_restores_old_version_and_keeps_current(store):
    a = store.save({"label": "v1", "memo": "m1"})
    store.save({"label": "v2", "memo": "m2"}, a["id"])
    r = store.revert(a["id"], 0)
    assert r["label"] == "v1" and r["memo"] == "m1"
    assert [h["label"] for h in r["history"]] == ["v1", "v2"]     # v2도 사라지지 않는다
    assert store.revert(a["id"], 99) is None


def test_update_keeps_other_fields_and_time(store):
    a = store.save({"label": "a", "memo": "", "snapshot": {"x": 1}})
    b = store.update(a["id"], {"memo": "m"})
    assert b["memo"] == "m" and b["label"] == "a" and b["snapshot"] == {"x": 1} and b["updated"] == a["updated"]
    assert store.update("nope", {"memo": "x"}) is None

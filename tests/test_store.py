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


def test_save_overwrite_delete(store):
    a = store.save({"label": "a"})
    b = store.save({"label": "b"})
    assert {p["label"] for p in store.list_all()} == {"a", "b"}
    a2 = store.save({"label": "a2"}, a["id"])                 # 덮어쓰기: id·created 유지
    assert a2["id"] == a["id"] and a2["created"] == a["created"]
    assert len(store.list_all()) == 2
    assert store.delete(b["id"]) and not store.delete(b["id"])
    assert [p["label"] for p in store.list_all()] == ["a2"]

"""HYROX 16구간. results 테이블의 split_* 컬럼과 1:1 대응 (초 단위)."""

SEGMENTS = [
    ("run1", "run"), ("ski_erg", "station"),
    ("run2", "run"), ("sled_push", "station"),
    ("run3", "run"), ("sled_pull", "station"),
    ("run4", "run"), ("burpee", "station"),
    ("run5", "run"), ("rowing", "station"),
    ("run6", "run"), ("farmers", "station"),
    ("run7", "run"), ("sandbag", "station"),
    ("run8", "run"), ("wall_balls", "station"),
]
NAMES = [n for n, _ in SEGMENTS]
KIND = dict(SEGMENTS)
COLUMNS = [f"split_{n}" for n in NAMES]
# total_time - (16구간 합) = Roxzone/전환 시간. 모델에서는 17번째 성분으로 다룬다.
OVERHEAD = "roxzone"
ALL = NAMES + [OVERHEAD]
GRIP_SEGMENTS = {"sled_pull", "farmers", "sandbag", "wall_balls"}

# race_id 접두어("season-8__HPRO_xxx" 의 HPRO) → 개인 모델에 쓸 수 있는 싱글 종목만
SINGLES = {"H": "open", "H1": "open", "HPRO": "pro", "HP1": "pro", "HP2": "pro"}

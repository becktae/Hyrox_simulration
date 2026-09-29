import os
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "Hyrox-data-agent" / "data" / "hyrox.db"
DB_PATH = Path(os.environ.get("HYROX_DB", DEFAULT_DB))
CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"

# total_time과 16구간 합의 차이(roxzone)가 이 범위를 벗어나면 데이터 오류로 보고 제외
OVERHEAD_RANGE = (100, 1500)

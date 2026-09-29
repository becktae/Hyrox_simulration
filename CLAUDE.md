# CLAUDE.md

HYROX 레이스 시뮬레이터 ("풋볼매니저식"). 수집된 선수의 실제 기록으로 개인 모델을 만들고, 전략을 골라 1~2분 레이스를 돌리거나 몬테카를로로 전략을 비교한다.

## Commands (이 디렉토리에서 실행)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.backtest --limit 150     # 예측 오차 측정 (튜닝 기준)
.venv/bin/uvicorn api.main:app --reload              # http://127.0.0.1:8000
```

## Architecture
- `sim/data.py` — `../Hyrox-data-agent/data/hyrox.db`를 **읽기 전용**으로 연다 (`HYROX_DB`로 변경). 절대 쓰지 않는다.
- `sim/profile.py` — 선수 프로필 = 개인 평균과 카테고리 평균의 shrinkage (`SHRINKAGE_K`).
- `sim/engine.py` — 16구간 엔진. 계획 페이스 기대값 == 개인 기준 기록이 되도록 피로를 "계획 궤적 대비 초과분"으로 계산.
- `sim/montecarlo.py`, `api/main.py`, `frontend/index.html` (번들러 없음).

## 데이터 함정 (중요)
- `athlete_id`는 사람 단위가 아니다(경기마다 다름). **(name, nationality)** 로 묶는다.
- `race_id` 접두어(`season-8__HPRO_...` 의 HPRO)로 종목 구분: H/HPRO 등 싱글만 개인 모델에 사용. HD/HDP(더블), HMR(믹스 릴레이)는 제외.
- 같은 사람의 기록이 서로 다른 race_id로 중복 저장된 경우가 있음(더블·프로 등) → race_id 기준 중복 제거.
- 구간 스플릿 약 30% 결측. 16구간이 모두 있는 경기만 프로필에 사용. 시간 단위는 초.
- `races.date`가 비어 있어 시간순은 race_id 문자열 정렬로 근사 (season 순서와 일치).

## 튜닝 규칙
엔진 계수를 바꾸면 반드시 `scripts.backtest`로 전후 비교. 목표 MAE ±120초.

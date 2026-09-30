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
- `sim/data.py` — `../Hyrox-data-agent/data/hyrox.db`를 **읽기 전용**으로 연다 (`HYROX_DB`로 변경). 절대 쓰지 않는다. `load_singles()`가 모델용 행(17성분)을 만든다.
- `sim/model.py` — 조건 효과 모델: `log x = 사람 + 조건(시즌,종목,성별) + 경험(N번째 경기) + 잔차`. 같은 사람이 여러 조건에서 뛴 쌍으로 시즌/무게 변화를 추정. 잔차에서 구간별 cv와 day_sd를 **데이터로** 추정. 결과는 `cache/model.json`에 저장(DB보다 오래되거나 `MODEL_VERSION`이 다르면 자동 재적합).
- `sim/profile.py` — 선수 프로필 = 본인 과거 경기를 조건·경험 보정해 목표 시즌 수준으로 환산한 평균 (+ 기록이 적을수록 넓어지는 불확실성).
- `sim/engine.py` — 16구간 + roxzone 엔진. 계획 페이스 기대값 == 개인 기준 기록. 페이스/피로/그립 계수는 **데이터로 보정되지 않은 설계값**.
- `sim/montecarlo.py`, `api/main.py`, `frontend/index.html` (번들러 없음).

## 데이터 함정 (중요)
- `athlete_id`는 사람 단위가 아니다(경기마다 다름). **(name, nationality)** 로 묶는다.
- `race_id` 접두어(`season-8__HPRO_...` 의 HPRO)로 종목 구분: H/HPRO 등 싱글만 사용. HD/HDP(더블), HMR(믹스 릴레이)는 제외.
- **같은 경기가 다른 race_id로 중복 저장**된다(예: `season-8__H_LR3...` 와 `season-8__H_SKI26_OVERALL`). 스플릿이 전부 같은 행은 제거(`load_singles`). 안 하면 백테스트에 누수가 생겨 성능이 크게 부풀려진다(실제로 한 번 겪음: MAE 83s → 실제 300s대).
- **`total_time` = 16구간 합 + roxzone(전환 시간)**. 구간 합만 예측하면 평균 ~400s 낮게 나온다. roxzone은 시즌별로 크게 다름(S6 ~350s, S7 ~400s, S8 ~520s).
- 구간 스플릿 약 30% 결측. 16구간이 모두 있고 roxzone이 100~1500s인 경기만 사용. 시간 단위는 초.
- `races.date`가 비어 있어 시간순은 race_id 문자열 정렬로 근사 (season 순서와 일치).
- 카테고리 평균(prior)으로 당기는 shrinkage는 **오히려 성능을 떨어뜨림**: 재출전 선수는 전체 참가자보다 훨씬 빠른 집단. 넣지 말 것.
- 데이터가 충분한 선수(2경기 이상, 온전한 스플릿)는 약 6천 명. 1경기뿐인 선수는 조건 보정과 경험 효과는 적용되지만 개인 추정이 불확실(구간이 넓어짐).

## 튜닝/검증 규칙
- `python -m scripts.backtest` (사람 단위 5-fold, 평가 대상은 모델 적합에서 제외). 지표는 P50 오차와 **P10~P90 캘리브레이션(목표 ≈80%)**.
- 현재(2026-09-30): MAE 321s, 편향 ~0s, ±120s 적중 32%, P10~P90 포함 85%. 같은 조건 MAE 279s.
- ±2분 평균 오차는 비현실적이다: 같은 선수의 경기 간 변동 자체가 크다. **공식 목표: P10~P90 포함률 80% ± 5%p** (전체/동일조건/조건변경 각각, `scripts/backtest.py`가 PASS/FAIL 표시). MAE는 참고 지표.
- `scripts/tune.py`, `scripts/tune_fe.py`: 모델 변형 비교(기준선/고정효과). 구조를 바꾸면 여기서 먼저 비교 후 backtest로 확정.

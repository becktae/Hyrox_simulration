# Hyrox_simulation

수집된 HYROX 기록(약 197k건)으로 특정 선수의 모델을 만들고 레이스를 시뮬레이션합니다.

1. 이름으로 선수 검색 → 2. 개인 프로필 생성 → 3. 페이스/전략 선택 → 4. 단일 레이스 또는 몬테카를로

상세 구조와 데이터 주의사항은 `CLAUDE.md` 참고. 실행: `uvicorn api.main:app --reload`

"""설정값. 판단 기준을 바꾸고 싶으면 이 파일만 고치면 됩니다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"          # 웹 대시보드 (GitHub Pages로 공개 가능)

# 주가 기록을 몇 거래일치 보관할지 (200일 이동평균 + 여유분)
HISTORY_DAYS = 260

# 등급 기준 (종합점수 0~100)
BUY_SCORE = 70     # 이 점수 이상 → 추천
SELL_SCORE = 40    # 이 점수 이하 → 비추천

# 최근 20일 평균 거래대금이 이보다 적으면 분석에서 제외 (거래가 거의 없는 종목)
MIN_TRADING_VALUE = {"KR": 100_000_000,   # 1억 원
                     "US": 1_000_000}     # 100만 달러

# 미국 재무지표(PER 등)는 종목마다 따로 받아야 해서 느립니다.
# 하루에 이만큼씩 갱신하고, 7일이 지난 것부터 새로 받습니다.
US_FUNDAMENTAL_PER_RUN = 1500
US_FUNDAMENTAL_MAX_AGE_DAYS = 7

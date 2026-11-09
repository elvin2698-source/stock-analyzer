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


def read_csv(path, **kw):
    """CSV 읽기. 'NA'(미국 종목코드) 같은 글자를 빈 값으로 착각하지 않도록 빈 칸만 빈 값으로 봅니다."""
    import pandas as pd
    return pd.read_csv(path, dtype={"ticker": str}, keep_default_na=False, na_values=[""], **kw)


# ── 모의투자 (균형형) ─────────────────────────────────────────
PAPER_DIR = ROOT / "paper"
PAPER_START_KRW = 50_000_000         # 시장별 가상 자금 (미국은 시작일 환율로 달러 환산)
PAPER_HOLDINGS = 20                  # 보유 종목 수
PAPER_SECTOR_CAP = 4                 # 업종 대분류당 최대 종목 수
PAPER_STOP_LOSS = -0.15              # 매수가 대비 -15%면 손절
PAPER_KEEP_RANK = 40                 # 주간 교체 때 종합점수 상위 40위 안이면 계속 보유
PAPER_SLIPPAGE = 0.001               # 체결 가격 불리하게 0.1% 반영
PAPER_COSTS = {                      # 수수료·세금 (대략적인 2026년 온라인 증권사 기준)
    "KR": {"buy": 0.00015, "sell": 0.00015 + 0.0020},   # 매도 시 증권거래세 0.20%
    "US": {"buy": 0.0025, "sell": 0.0025},               # 해외주식 수수료 0.25%
}
PAPER_BENCHMARK = {"KR": ("^KS11", "코스피"), "US": ("^GSPC", "S&P500")}

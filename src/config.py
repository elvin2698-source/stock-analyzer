"""설정값. 판단 기준을 바꾸고 싶으면 이 파일만 고치면 됩니다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"          # 웹 대시보드 (GitHub Pages로 공개 가능)

# 주가 기록을 몇 거래일치 보관할지 (약 2년: 200일 이동평균 + 백테스트용)
HISTORY_DAYS = 520

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


# ── 모의투자 ──────────────────────────────────────────────
# 규칙을 바꾸면 PAPER_RULES_VERSION 을 올리고 paper/RULES.md 에 바뀐 내용을 적습니다.
PAPER_RULES_VERSION = "v2"
PAPER_DIR = ROOT / "paper"
PAPER_START_KRW = 50_000_000         # 계좌별 가상 자금 (미국 계좌는 시작일 환율로 달러 환전)
PAPER_SLIPPAGE = 0.001               # 체결 가격 불리하게 0.1%
PAPER_MAX_ADV = 0.01                 # 한 번에 살 수 있는 금액: 최근 20일 평균 거래대금의 1%까지
PAPER_LIMIT_KR = 0.295               # 한국 상·하한가(±30%) 근처면 체결 안 됨
PAPER_FX_COST = 0.0025               # 원→달러 환전 수수료 0.25% (계좌 개설 때 한 번)
PAPER_COSTS = {                      # 수수료·세금 (대략적인 2026년 온라인 증권사 기준)
    "KR": {"buy": 0.00015, "sell": 0.00015 + 0.0020},   # 매도 시 증권거래세 0.20%
    "US": {"buy": 0.0025, "sell": 0.0025},               # 해외주식 수수료 0.25%
}
PAPER_DIV_TAX = {"KR": 0.154, "US": 0.15}            # 배당소득세 원천징수
PAPER_US_CGT = (0.22, 2_500_000)     # 해외주식 양도소득세 22%, 연 250만원 공제 (추정치로만 표시)
PAPER_RF = 0.03                      # 샤프지수용 무위험 수익률 (연 3%)
PAPER_REGIME_INDEX = {"KR": "^KS11", "US": "^GSPC"}  # 하락장 판단 지수 (200일선 아래면 하락장)
PAPER_BENCHMARKS = {
    "KR": [("^KS11", "코스피"), ("^KQ11", "코스닥")],
    "US": [("^GSPC", "S&P500"), ("^SP500TR", "S&P500 배당포함"), ("^RUT", "러셀2000(소형주)")],
}
# 전략 계좌. rebalance: weekly(매주 첫 분석) / monthly(매달 첫 분석) / daily
PAPER_STRATEGIES = {
    "balanced":   {"name": "균형형", "n": 20, "sector_cap": 4, "stop": -0.15, "keep_rank": 40,
                   "rebalance": "weekly", "sizes": None, "pick": "top", "regime": False},
    "stable":     {"name": "안정형", "n": 20, "sector_cap": 4, "stop": -0.10, "keep_rank": 40,
                   "rebalance": "monthly", "sizes": ["대형주", "중형주"], "pick": "top", "regime": False,
                   "min_score": 60},       # 대·중형주는 70점(추천)을 넘는 종목이 드물어 60점 이상에서 고름
    "aggressive": {"name": "공격형", "n": 10, "sector_cap": 3, "stop": -0.20, "keep_rank": 20,
                   "rebalance": "daily", "sizes": None, "pick": "top", "regime": False},
    "defensive":  {"name": "균형형+하락장 방어", "n": 20, "sector_cap": 4, "stop": -0.15, "keep_rank": 40,
                   "rebalance": "weekly", "sizes": None, "pick": "top", "regime": True},
    "random":     {"name": "무작위 대조군", "n": 20, "sector_cap": 4, "stop": -0.15, "keep_rank": 40,
                   "rebalance": "weekly", "sizes": None, "pick": "random", "regime": False},
}

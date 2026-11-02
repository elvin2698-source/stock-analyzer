"""한국 주식(코스피·코스닥) 전 종목 데이터 수집. 출처: KRX 정보데이터시스템 (pykrx)

KRX는 무료 회원 로그인이 필요합니다. 환경변수 KRX_ID, KRX_PW 를 설정하세요.
"""
import time
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock

from config import DATA_DIR, HISTORY_DAYS

PRICES = DATA_DIR / "prices_kr.csv"
COMPANIES = DATA_DIR / "companies_kr.csv"

# 분석에 필요한 열만 저장 (파일 크기 절약)
OHLCV_COLS = {"종가": "close", "거래량": "volume", "등락률": "change_pct"}


def _ymd(d):
    return d.strftime("%Y%m%d")


def _fetch_day(day: str) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_ticker(day, market="ALL")
    if df is None or df.empty or df["거래량"].sum() == 0:
        return pd.DataFrame()           # 휴장일
    df = df.rename(columns=OHLCV_COLS)
    df = df[[c for c in OHLCV_COLS.values() if c in df.columns]]
    df.index.name = "ticker"
    df = df.reset_index()
    df.insert(0, "date", pd.Timestamp(day).date().isoformat())
    return df


def update_prices(today: str) -> pd.DataFrame:
    old = pd.read_csv(PRICES, dtype={"ticker": str}) if PRICES.exists() else pd.DataFrame()
    end = datetime.strptime(today, "%Y%m%d")
    if old.empty:
        start = end - timedelta(days=int(HISTORY_DAYS * 1.5))   # 거래일 260일 ≈ 달력 390일
    else:
        start = datetime.strptime(old["date"].max(), "%Y-%m-%d") + timedelta(days=1)

    frames, d = [], start
    while d <= end:
        if d.weekday() < 5:
            day = _fetch_day(_ymd(d))
            if not day.empty:
                frames.append(day)
                print(f"  [KR] {d.date()} {len(day)}종목")
            time.sleep(0.3)
        d += timedelta(days=1)

    new = pd.concat([old, *frames], ignore_index=True) if frames else old
    new = new.drop_duplicates(["date", "ticker"], keep="last")
    keep = sorted(new["date"].unique())[-HISTORY_DAYS:]
    new = new[new["date"].isin(keep)].sort_values(["date", "ticker"])
    new.to_csv(PRICES, index=False)
    return new


def update_companies(today: str) -> pd.DataFrame:
    rows = []
    for market in ("KOSPI", "KOSDAQ"):
        sec = stock.get_market_sector_classifications(today, market)
        sec = sec.rename(columns={"종목명": "name", "업종명": "sector"})[["name", "sector"]]
        sec["market"] = market
        rows.append(sec)
    comp = pd.concat(rows)
    comp.index.name = "ticker"

    fund = stock.get_market_fundamental_by_ticker(today, market="ALL")
    fund = fund.rename(columns={"PER": "per", "PBR": "pbr", "EPS": "eps",
                                "BPS": "bps", "DIV": "div_yield", "DPS": "dps"})
    cap = stock.get_market_cap_by_ticker(today, market="ALL")[["시가총액"]].rename(columns={"시가총액": "market_cap"})

    comp = comp.join(fund[["per", "pbr", "eps", "bps", "div_yield"]], how="left").join(cap, how="left")
    # 0은 'PER 계산 불가(적자 등)'라는 뜻 → 빈 값으로
    for c in ("per", "pbr"):
        comp.loc[comp[c] <= 0, c] = float("nan")
    comp["roe"] = (comp["eps"] / comp["bps"] * 100).where(comp["bps"] > 0)   # ROE ≈ EPS / BPS
    comp["updated"] = pd.Timestamp(today).date().isoformat()
    comp = comp.reset_index()
    comp.to_csv(COMPANIES, index=False)
    return comp


def run():
    today = stock.get_nearest_business_day_in_a_week()
    print(f"[KR] 기준일 {today}")
    comp = update_companies(today)
    print(f"[KR] 기업 정보 {len(comp)}개")
    prices = update_prices(today)
    print(f"[KR] 주가 {prices['date'].nunique()}거래일 보관")


if __name__ == "__main__":
    run()

"""미국 주식(나스닥·뉴욕·아멕스) 전 종목 데이터 수집.
종목 목록: Nasdaq 스크리너 / 주가·재무: Yahoo Finance (yfinance)
"""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf

from config import DATA_DIR, HISTORY_DAYS, US_FUNDAMENTAL_PER_RUN, US_FUNDAMENTAL_MAX_AGE_DAYS

PRICES = DATA_DIR / "prices_us.csv"
COMPANIES = DATA_DIR / "companies_us.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_listing() -> pd.DataFrame:
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true"
    rows = requests.get(url, headers=UA, timeout=60).json()["data"]["rows"]
    df = pd.DataFrame(rows)
    df = df.rename(columns={"symbol": "ticker", "marketCap": "market_cap"})
    df["ticker"] = df["ticker"].str.strip()
    # 우선주·워런트·유닛 등 제외 (기호가 들어간 티커)
    df = df[df["ticker"].str.fullmatch(r"[A-Z]{1,5}")]
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
    df = df[df["market_cap"] > 0]
    return df[["ticker", "name", "sector", "industry", "country", "market_cap"]]


def update_prices(tickers) -> pd.DataFrame:
    old = pd.read_csv(PRICES) if PRICES.exists() else pd.DataFrame()
    start = (date.today() - timedelta(days=int(HISTORY_DAYS * 1.5))) if old.empty \
        else date.fromisoformat(old["date"].max()) - timedelta(days=5)   # 며칠 겹치게 받아 수정주가 반영

    frames, tickers = [], list(tickers)
    for i in range(0, len(tickers), 200):
        batch = tickers[i:i + 200]
        raw = yf.download(batch, start=start.isoformat(), auto_adjust=True,
                          group_by="column", progress=False, threads=True)
        if raw.empty:
            continue
        long = raw.stack(level=1, future_stack=True).reset_index()
        long.columns = [str(c).lower() for c in long.columns]
        long = long.rename(columns={"price": "date", "level_1": "ticker"})
        long = long.dropna(subset=["close"])
        long["date"] = pd.to_datetime(long["date"]).dt.date.astype(str)
        frames.append(long[["date", "ticker", "close", "volume"]])
        print(f"  [US] 주가 {min(i + 200, len(tickers))}/{len(tickers)}")
        time.sleep(1)

    new = pd.concat([old, *frames], ignore_index=True) if frames else old
    new = new.drop_duplicates(["date", "ticker"], keep="last")
    keep = sorted(new["date"].unique())[-HISTORY_DAYS:]
    new = new[new["date"].isin(keep)].sort_values(["date", "ticker"])
    new["volume"] = new["volume"].fillna(0).astype("int64")
    new.to_csv(PRICES, index=False, float_format="%.4f")
    return new


def _fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    pe, pb = info.get("trailingPE"), info.get("priceToBook")
    roe, dy = info.get("returnOnEquity"), info.get("dividendYield")
    return {"ticker": ticker,
            "per": pe if pe and pe > 0 else None,
            "pbr": pb if pb and pb > 0 else None,
            "roe": roe * 100 if roe is not None else None,
            # yfinance 버전에 따라 0.02 또는 2.0 형식 → %로 통일
            "div_yield": (dy * 100 if dy is not None and dy < 1 else dy),
            "fund_updated": date.today().isoformat()}


def update_companies() -> pd.DataFrame:
    listing = fetch_listing()
    old = pd.read_csv(COMPANIES) if COMPANIES.exists() else pd.DataFrame(columns=["ticker"])
    fund_cols = ["per", "pbr", "roe", "div_yield", "fund_updated"]
    comp = listing.merge(old[["ticker", *[c for c in fund_cols if c in old]]], on="ticker", how="left")

    # 오래된 재무정보부터, 시가총액 큰 순으로 일부만 갱신
    if "fund_updated" not in comp:
        comp["fund_updated"] = None
    age = (pd.Timestamp.today() - pd.to_datetime(comp["fund_updated"])).dt.days.fillna(9999)
    todo = comp[age >= US_FUNDAMENTAL_MAX_AGE_DAYS].sort_values("market_cap", ascending=False)
    todo = todo["ticker"].head(US_FUNDAMENTAL_PER_RUN).tolist()
    print(f"[US] 재무지표 갱신 {len(todo)}종목")
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = [r for r in ex.map(_fundamentals, todo) if r]
    if got:
        upd = pd.DataFrame(got).set_index("ticker")
        comp = comp.set_index("ticker")
        for c in fund_cols:
            if c not in comp:
                comp[c] = None
        comp.update(upd)
        comp = comp.reset_index()

    comp["market"] = "US"
    comp.to_csv(COMPANIES, index=False)
    return comp


def run():
    comp = update_companies()
    print(f"[US] 기업 정보 {len(comp)}개")
    prices = update_prices(comp["ticker"])
    print(f"[US] 주가 {prices['date'].nunique()}거래일 보관")


if __name__ == "__main__":
    run()

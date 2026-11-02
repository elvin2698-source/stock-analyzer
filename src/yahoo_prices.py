"""Yahoo Finance에서 여러 종목의 일별 종가·거래량을 받아 CSV에 이어 붙입니다 (한국·미국 공용)."""
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from config import HISTORY_DAYS


def update_prices(path, symbols: dict, label: str) -> pd.DataFrame:
    """symbols: {저장할 종목코드: 야후 심볼}  예) {"005930": "005930.KS"}, {"AAPL": "AAPL"}"""
    old = pd.read_csv(path, dtype={"ticker": str}) if path.exists() else pd.DataFrame()
    if old.empty:
        start = date.today() - timedelta(days=int(HISTORY_DAYS * 1.5))   # 거래일 260일 ≈ 달력 390일
    else:
        start = date.fromisoformat(old["date"].max()) - timedelta(days=5)  # 며칠 겹쳐 받아 수정주가 반영

    back = {v: k for k, v in symbols.items()}
    yahoo = list(symbols.values())
    frames = []
    for i in range(0, len(yahoo), 200):
        batch = yahoo[i:i + 200]
        raw = yf.download(batch, start=start.isoformat(), auto_adjust=True,
                          group_by="column", progress=False, threads=True)
        if raw.empty:
            continue
        if not isinstance(raw.columns, pd.MultiIndex):        # 1종목만 받은 경우
            raw.columns = pd.MultiIndex.from_product([raw.columns, batch])
        long = raw[["Close", "Volume"]].stack(level=1, future_stack=True).reset_index()
        long.columns = ["date", "symbol", "close", "volume"]
        long = long.dropna(subset=["close"])
        long["date"] = pd.to_datetime(long["date"]).dt.date.astype(str)
        long["ticker"] = long["symbol"].map(back)
        frames.append(long[["date", "ticker", "close", "volume"]])
        print(f"  [{label}] 주가 {min(i + 200, len(yahoo))}/{len(yahoo)}")
        time.sleep(1)

    if not frames and old.empty:
        raise RuntimeError(f"[{label}] 주가를 하나도 받지 못했습니다 (야후 접속 실패?)")
    new = pd.concat([old, *frames], ignore_index=True)
    new = new.drop_duplicates(["date", "ticker"], keep="last")
    keep = sorted(new["date"].unique())[-HISTORY_DAYS:]
    new = new[new["date"].isin(keep)].sort_values(["date", "ticker"])
    new["volume"] = new["volume"].fillna(0).astype("int64")
    new.to_csv(path, index=False, float_format="%.4f")
    return new

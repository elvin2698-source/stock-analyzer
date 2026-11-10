"""Yahoo Finance에서 여러 종목의 일별 시가·종가·거래량·배당·액면분할을 받아 CSV(gzip)에 이어 붙입니다.

저장 형식: date, ticker, open, close, volume, dividend, split
- close/open 은 Yahoo 관례대로 '액면분할만 반영된' 가격입니다 (배당은 반영 안 됨).
- 새로 받은 데이터에 액면분할이 있으면, 이전에 저장해 둔 그 종목의 과거 가격·거래량도 같은 비율로 고칩니다.
- 배당까지 반영한 수정주가는 analyze.adjusted_close() 가 이 파일로 직접 계산합니다.
"""
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from config import HISTORY_DAYS, read_csv

COLS = ["date", "ticker", "open", "close", "volume", "dividend", "split"]


def _download(batch, start):
    raw = yf.download(batch, start=start.isoformat(), auto_adjust=False, actions=True,
                      group_by="column", progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame(columns=COLS + ["symbol"])
    if not isinstance(raw.columns, pd.MultiIndex):        # 1종목만 받은 경우
        raw.columns = pd.MultiIndex.from_product([raw.columns, batch])
    for c in ("Dividends", "Stock Splits"):
        if c not in raw.columns.get_level_values(0):
            raw[[(c, s) for s in batch]] = 0.0
    long = raw[["Open", "Close", "Volume", "Dividends", "Stock Splits"]].stack(level=1, future_stack=True).reset_index()
    long.columns = ["date", "symbol", "open", "close", "volume", "dividend", "split"]
    long = long.dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.date.astype(str)
    long[["dividend", "split"]] = long[["dividend", "split"]].fillna(0.0)
    return long


def _apply_new_splits(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """새 데이터에서 처음 보는 액면분할이면, 그 날짜 이전에 저장된 가격·거래량을 같은 비율로 고친다."""
    if old.empty:
        return old
    ev = new[new["split"] > 0][["ticker", "date", "split"]]
    if ev.empty:
        return old
    seen = set(zip(old.loc[old["split"] > 0, "ticker"], old.loc[old["split"] > 0, "date"]))
    old = old.copy()
    for t, d, r in ev.itertuples(index=False):
        if (t, d) in seen:
            continue                                         # 이미 반영한 분할
        m = (old["ticker"] == t) & (old["date"] < d)
        old.loc[m, ["open", "close"]] = old.loc[m, ["open", "close"]] / r
        old.loc[m, "volume"] = old.loc[m, "volume"] * r
        old.loc[m, "dividend"] = old.loc[m, "dividend"] / r
        print(f"  액면분할 반영: {t} {d} ×{r:g}")
    return old


def update_prices(path, symbols: dict, label: str) -> pd.DataFrame:
    """symbols: {저장할 종목코드: 야후 심볼}  예) {"005930": "005930.KS"}, {"AAPL": "AAPL"}"""
    old = read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    if old.empty:
        start = date.today() - timedelta(days=int(HISTORY_DAYS * 1.5))   # 거래일 → 달력일
    else:
        start = date.fromisoformat(old["date"].max()) - timedelta(days=7)  # 며칠 겹쳐 받아 빠진 날 보충

    back = {v: k for k, v in symbols.items()}
    yahoo = list(symbols.values())
    frames = []
    for i in range(0, len(yahoo), 200):
        batch = yahoo[i:i + 200]
        long = _download(batch, start)
        if long.empty:
            continue
        long["ticker"] = long["symbol"].map(back)
        frames.append(long[COLS])
        print(f"  [{label}] 주가 {min(i + 200, len(yahoo))}/{len(yahoo)}")
        time.sleep(1)

    if not frames and old.empty:
        raise RuntimeError(f"[{label}] 주가를 하나도 받지 못했습니다 (야후 접속 실패?)")
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
    old = _apply_new_splits(old, new)
    allp = pd.concat([old, new], ignore_index=True) if not old.empty else new
    allp = allp.drop_duplicates(["date", "ticker"], keep="last")
    keep = sorted(allp["date"].unique())[-HISTORY_DAYS:]
    allp = allp[allp["date"].isin(keep)].sort_values(["date", "ticker"])
    allp["volume"] = allp["volume"].fillna(0).astype("int64")
    allp.to_csv(path, index=False, float_format="%.6g", compression="gzip")
    return allp

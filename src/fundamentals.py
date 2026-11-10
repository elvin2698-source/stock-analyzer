"""과거 시점 재무 데이터 (백테스트용 PER·PBR·ROE를 날짜별로 다시 계산하기 위한 재료).

- 한국: 네이버 증권 연간(최근 3~4년)·분기(최근 5분기) EPS·BPS
- 미국: 야후 연간 재무제표(최근 4~5년) 희석 EPS, 주당순자산(자본 ÷ 주식 수)
- 실적은 결산 뒤 몇 달 지나야 발표되므로 '쓸 수 있게 된 날(available)'을 둡니다.
    한국 연간 90일·분기 45일 / 미국 연간 75일 뒤부터 사용 (법정 제출 기한 기준, 미래 정보 엿보기 방지)
- 날짜 d의 EPS: 그때까지 발표된 최근 4개 분기 합(있으면) → 없으면 최근 연간 EPS
  날짜 d의 BPS: 그때까지 발표된 가장 최근 값
저장: data/fundamentals_kr.csv, data/fundamentals_us.csv
"""
import calendar
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from config import DATA_DIR, read_csv

PATH = {"KR": DATA_DIR / "fundamentals_kr.csv", "US": DATA_DIR / "fundamentals_us.csv"}
COLS = ["ticker", "period", "kind", "eps", "bps", "available", "fetched"]
LAG = {"KR": {"A": 90, "Q": 45}, "US": {"A": 75, "Q": 45}}
REFRESH_DAYS = {"KR": 30, "US": 90}
PER_RUN = {"KR": 4000, "US": 1500}
UA = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"}


def _num(s):
    s = str(s or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _month_end(key: str) -> date:
    y, m = int(key[:4]), int(key[4:6])
    return date(y, m, calendar.monthrange(y, m)[1])


def _rows(ticker, period: date, kind, eps, bps, market):
    avail = (period + timedelta(days=LAG[market][kind])).isoformat()
    return {"ticker": ticker, "period": period.isoformat(), "kind": kind, "eps": eps, "bps": bps,
            "available": avail, "fetched": date.today().isoformat()}


def fetch_kr_one(code: str):
    out = []
    for kind, path in (("A", "annual"), ("Q", "quarter")):
        try:
            r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/finance/{path}", headers=UA, timeout=15)
            fi = r.json()["financeInfo"]
        except Exception:
            continue
        rows = {x["title"]: x["columns"] for x in fi.get("rowList", [])}
        for t in fi.get("trTitleList", []):
            if t.get("isConsensus") != "N":
                continue                                    # 증권사 예상치는 쓰지 않음
            k = t["key"]
            eps = _num(rows.get("EPS", {}).get(k, {}).get("value"))
            bps = _num(rows.get("BPS", {}).get(k, {}).get("value"))
            if np.isnan(eps) and np.isnan(bps):
                continue
            out.append(_rows(code, _month_end(k), kind, eps, bps, "KR"))
    return out


def fetch_us_one(sym: str):
    import yfinance as yf
    try:
        t = yf.Ticker(sym.replace(".", "-"))
        inc, bal = t.income_stmt, t.balance_sheet
    except Exception:
        return []
    if inc is None or inc.empty or "Diluted EPS" not in inc.index:
        return []
    out = []
    for col in inc.columns:
        eps = inc.at["Diluted EPS", col]
        if pd.isna(eps):
            continue
        bps = np.nan
        if bal is not None and not bal.empty and col in bal.columns and "Common Stock Equity" in bal.index:
            eq = bal.at["Common Stock Equity", col]
            # 주식 수는 'EPS 기준'으로 맞춤(순이익 ÷ EPS) → 액면분할이 있어도 EPS와 같은 기준
            ni = inc.at["Net Income Common Stockholders", col] if "Net Income Common Stockholders" in inc.index else np.nan
            sh = ni / eps if pd.notna(ni) and eps and abs(eps) > 1e-9 else np.nan
            if (pd.isna(sh) or sh <= 0) and "Ordinary Shares Number" in bal.index:
                sh = bal.at["Ordinary Shares Number", col]
            if pd.notna(eq) and pd.notna(sh) and sh > 0:
                bps = float(eq) / float(sh)
        out.append(_rows(sym, pd.Timestamp(col).date(), "A", float(eps), bps, "US"))
    return out


def update(market: str, tickers: list):
    """tickers: 우선순위 순서(시가총액 큰 순). 오래된 것부터 정해진 개수만큼 새로 받음."""
    path = PATH[market]
    old = read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)
    last = old.groupby("ticker")["fetched"].max() if len(old) else pd.Series(dtype=object)
    today = pd.Timestamp.today()
    age = pd.Series({t: (today - pd.Timestamp(last[t])).days if t in last.index else 9999 for t in tickers})
    todo = [t for t in tickers if age[t] >= REFRESH_DAYS[market]][:PER_RUN[market]]
    if not todo:
        print(f"  [재무 이력] {market} 새로 받을 종목 없음 ({len(last)}종목 보유)")
        return old
    fn = fetch_kr_one if market == "KR" else fetch_us_one
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8 if market == "KR" else 4) as ex:
        got = [r for rows in ex.map(fn, todo) for r in rows]
    new = pd.DataFrame(got, columns=COLS)
    done = set(new["ticker"])
    keep = old[~old["ticker"].isin(done)]
    out = pd.concat([keep, new], ignore_index=True).sort_values(["ticker", "period", "kind"])
    out.to_csv(path, index=False)
    print(f"  [재무 이력] {market} {len(done)}/{len(todo)}종목 받음 ({time.time() - t0:.0f}초) · 전체 {out['ticker'].nunique()}종목")
    return out


def load(market: str) -> pd.DataFrame:
    p = PATH[market]
    return read_csv(p) if p.exists() else pd.DataFrame(columns=COLS)


def at(fund: pd.DataFrame, d: str) -> pd.DataFrame:
    """날짜 d에 '이미 발표돼 있던' 실적만으로 종목별 EPS(최근 4분기 합 또는 연간)·BPS."""
    f = fund[fund["available"] <= d]
    if f.empty:
        return pd.DataFrame(columns=["eps", "bps"])
    a = f[f["kind"] == "A"].sort_values("period").groupby("ticker").last()
    q = f[f["kind"] == "Q"].sort_values("period")
    q4 = q.groupby("ticker").tail(4)
    g = q4.groupby("ticker")
    span = (pd.to_datetime(g["period"].max()) - pd.to_datetime(g["period"].min())).dt.days
    ttm = g["eps"].sum(min_count=4).where((g["eps"].count() == 4) & (span <= 300))
    if not len(a):
        eps = ttm
    elif not len(ttm.dropna()):
        eps = a["eps"]
    else:
        eps = ttm.combine_first(a["eps"])
    b = f.dropna(subset=["bps"]).sort_values("period").groupby("ticker")["bps"].last()
    return pd.DataFrame({"eps": eps, "bps": b})

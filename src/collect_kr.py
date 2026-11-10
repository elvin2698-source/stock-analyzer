"""한국 주식(코스피·코스닥) 전 종목 데이터 수집. 로그인이 필요 없는 공개 자료만 씁니다.

- 종목 목록·현재가·시가총액   : 네이버 증권 모바일 API (시가총액 순위)
- PER·PBR·EPS·BPS·배당수익률  : 네이버 증권 모바일 API (종목별 정보)
- 업종                        : KRX 상장공시시스템(KIND) 상장법인 목록
- 일별 주가                   : Yahoo Finance (005930.KS / 035720.KQ 형식)
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from io import StringIO

import pandas as pd
import requests

from config import DATA_DIR
from yahoo_prices import update_prices

PRICES = DATA_DIR / "prices_kr.v2.csv.gz"
COMPANIES = DATA_DIR / "companies_kr.csv"
UA = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"}
API = "https://m.stock.naver.com/api"
MARKETS = {"KOSPI": "KS", "KOSDAQ": "KQ"}


def _get(url, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def _num(s):
    """'10.32배', '22,292원', '0.73%', 'N/A' → 숫자 (없으면 NaN)"""
    m = re.search(r"-?[\d,]+(?:\.\d+)?", str(s or ""))
    return float(m.group(0).replace(",", "")) if m else float("nan")


def fetch_listing(market: str) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        j = _get(f"{API}/stocks/marketValue/{market}?page={page}&pageSize=100")
        rows += j["stocks"]
        if page * 100 >= j["totalCount"] or not j["stocks"]:
            break
        page += 1
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df = df[df["stockEndType"] == "stock"]            # ETF·ETN 등 제외
    out = pd.DataFrame({
        "ticker": df["itemCode"], "name": df["stockName"], "market": market,
        "market_cap": pd.to_numeric(df["marketValueRaw"], errors="coerce"),
        "halted": df["tradeStopType"].map(lambda x: (x or {}).get("name", "TRADING") != "TRADING"),
    })
    print(f"  [KR] {market} 종목 {len(out)}개")
    return out


def _fundamentals(ticker):
    try:
        info = {x["code"]: x["value"] for x in _get(f"{API}/stock/{ticker}/integration")["totalInfos"]}
    except Exception:
        return None
    eps, bps = _num(info.get("eps")), _num(info.get("bps"))
    return {"ticker": ticker, "per": _num(info.get("per")), "pbr": _num(info.get("pbr")),
            "eps": eps, "bps": bps, "div_yield": _num(info.get("dividendYieldRatio"))}


def fetch_sectors() -> pd.DataFrame:
    """KIND 상장법인 목록 (종목코드 → 업종). 실패해도 분석은 계속합니다."""
    try:
        r = requests.get("https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13",
                         headers=UA, timeout=60)
        r.encoding = "euc-kr"
        t = pd.read_html(StringIO(r.text), header=0, converters={"종목코드": str})[0]
        t["ticker"] = t["종목코드"].str.strip().str.zfill(6)
        return t[["ticker", "업종"]].rename(columns={"업종": "sector"})
    except Exception as e:
        print(f"  [KR] 업종 정보 받기 실패 ({type(e).__name__}) → 시장 전체 기준으로 비교")
        return pd.DataFrame(columns=["ticker", "sector"])


def update_companies() -> pd.DataFrame:
    comp = pd.concat([fetch_listing(m) for m in MARKETS], ignore_index=True).drop_duplicates("ticker")

    with ThreadPoolExecutor(max_workers=8) as ex:
        fund = [f for f in ex.map(_fundamentals, comp["ticker"]) if f]
    print(f"  [KR] 재무지표 {len(fund)}/{len(comp)}종목")
    comp = comp.merge(pd.DataFrame(fund), on="ticker", how="left")

    comp = comp.merge(fetch_sectors(), on="ticker", how="left")
    common = comp["ticker"].str[:5] + "0"                      # 우선주(005935) → 보통주(005930) 업종 사용
    comp["sector"] = comp["sector"].fillna(common.map(comp.set_index("ticker")["sector"]))
    comp["sector"] = comp["sector"].fillna(comp["market"])
    for c in ("per", "pbr"):
        comp.loc[comp[c] <= 0, c] = float("nan")
    comp["roe"] = (comp["eps"] / comp["bps"] * 100).where(comp["bps"] > 0)   # ROE ≈ EPS / BPS
    comp["yahoo"] = comp["ticker"] + "." + comp["market"].map(MARKETS)
    comp["updated"] = date.today().isoformat()
    comp.to_csv(COMPANIES, index=False)
    return comp


def run():
    comp = update_companies()
    print(f"[KR] 기업 정보 {len(comp)}개")
    prices = update_prices(PRICES, dict(zip(comp["ticker"], comp["yahoo"])), "KR")
    print(f"[KR] 주가 {prices['date'].nunique()}거래일 보관")


if __name__ == "__main__":
    run()

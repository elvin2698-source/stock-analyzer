"""한국 주식(코스피·코스닥) 전 종목 데이터 수집. 로그인이 필요 없는 공개 자료만 씁니다.

- 종목 목록·현재가·시가총액·PER·ROE : 네이버 금융 시가총액 순위 페이지
- 업종                               : KRX 상장공시시스템(KIND) 상장법인 목록
- 일별 주가                          : Yahoo Finance (005930.KS / 035720.KQ 형식)
"""
import re
import time
from datetime import date
from io import StringIO

import pandas as pd
import requests

from config import DATA_DIR
from yahoo_prices import update_prices

PRICES = DATA_DIR / "prices_kr.csv"
COMPANIES = DATA_DIR / "companies_kr.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
NAVER = "https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
MARKETS = {"KOSPI": (0, "KS"), "KOSDAQ": (1, "KQ")}


def _clean(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _num(s):
    s = str(s).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _naver_page(sosok: int, page: int):
    r = requests.get(NAVER.format(sosok=sosok, page=page), headers=UA, timeout=30)
    r.encoding = "euc-kr"
    html = r.text
    table = re.search(r'<table[^>]*class="type_2".*?</table>', html, re.S).group(0)
    headers = [_clean(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", table, re.S)]
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S):
        m = re.search(r'code=([0-9A-Z]{6})"[^>]*>(.*?)</a>', tr)
        if not m:
            continue
        cells = [_clean(td) for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        row = dict(zip(headers, cells))
        row["ticker"], row["name"] = m.group(1), _clean(m.group(2))
        rows.append(row)
    last = re.search(r'class="pgRR".*?page=(\d+)', html, re.S)
    return rows, int(last.group(1)) if last else page


def fetch_naver(market: str) -> pd.DataFrame:
    sosok = MARKETS[market][0]
    rows, last = _naver_page(sosok, 1)
    for p in range(2, last + 1):
        more, _ = _naver_page(sosok, p)
        rows += more
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    out = pd.DataFrame({
        "ticker": df["ticker"], "name": df["name"], "market": market,
        "price_naver": df["현재가"].map(_num),
        "market_cap": df["시가총액"].map(_num) * 1e8,     # 단위: 억원 → 원
        "per": df["PER"].map(_num),
        "roe": df["ROE"].map(_num),
    })
    print(f"  [KR] 네이버 {market} {len(out)}종목")
    return out


def fetch_sectors() -> pd.DataFrame:
    """KIND 상장법인 목록 (종목코드 → 업종). 실패해도 분석은 계속합니다."""
    try:
        r = requests.get("https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13",
                         headers=UA, timeout=60)
        r.encoding = "euc-kr"
        t = pd.read_html(StringIO(r.text), header=0)[0]
        t["ticker"] = t["종목코드"].astype(str).str.zfill(6)
        return t[["ticker", "업종"]].rename(columns={"업종": "sector"})
    except Exception as e:
        print(f"  [KR] 업종 정보 받기 실패 ({type(e).__name__}) → 시장 전체 기준으로 비교")
        return pd.DataFrame(columns=["ticker", "sector"])


def update_companies() -> pd.DataFrame:
    comp = pd.concat([fetch_naver(m) for m in MARKETS], ignore_index=True)
    comp = comp.drop_duplicates("ticker")
    comp = comp.merge(fetch_sectors(), on="ticker", how="left")
    comp["sector"] = comp["sector"].fillna(comp["market"])
    comp.loc[comp["per"] <= 0, "per"] = float("nan")
    # PBR = 주가/순자산 = (주가/이익) × (이익/순자산) = PER × ROE
    comp["pbr"] = (comp["per"] * comp["roe"] / 100).where((comp["per"] > 0) & (comp["roe"] > 0))
    comp["div_yield"] = float("nan")      # 무료 공개 자료로는 전 종목 배당을 받기 어려움 → 중립 처리
    comp["yahoo"] = comp["ticker"] + "." + comp["market"].map({m: s for m, (_, s) in MARKETS.items()})
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

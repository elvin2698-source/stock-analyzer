"""규칙 기반 모의투자 (균형형). 분석이 끝날 때마다 시장별 가상 계좌를 갱신합니다.

매매 규칙
  매일   : 매수가 대비 -15% 이하로 떨어지거나 등급이 '비추천'이 되면 판다.
  주 1회 : (그 주의 첫 분석 때) 종합점수 상위 40위 밖으로 밀린 종목을 팔고,
           '추천' 등급 중 점수 높은 순으로 빈자리를 채운다. 업종 대분류당 최대 4종목,
           종목당 계좌 평가금액의 1/20씩 같은 금액으로 산다.
  체결   : 그날 종가에 0.1% 불리하게 + 수수료·세금 반영. 소수점 주식은 사지 않는다.

저장 파일 (paper/)
  state.json   계좌 현금·시작 정보
  holdings.csv 보유 종목
  trades.csv   매매 내역
  history.csv  날짜별 평가금액과 시장 지수 비교
"""
import json
import math

import pandas as pd

from config import (PAPER_DIR, PAPER_START_KRW, PAPER_HOLDINGS, PAPER_SECTOR_CAP, PAPER_STOP_LOSS,
                    PAPER_KEEP_RANK, PAPER_SLIPPAGE, PAPER_COSTS, PAPER_BENCHMARK, read_csv)

STATE = PAPER_DIR / "state.json"
HOLD = PAPER_DIR / "holdings.csv"
TRADES = PAPER_DIR / "trades.csv"
HIST = PAPER_DIR / "history.csv"
HOLD_COLS = ["account", "ticker", "name", "group", "shares", "avg_cost", "buy_date"]
TRADE_COLS = ["date", "account", "side", "ticker", "name", "shares", "price", "amount", "cost", "pnl", "pnl_pct", "reason"]
HIST_COLS = ["date", "account", "cash", "stock_value", "nav", "return_pct", "bench", "bench_return_pct", "holdings"]


def _load():
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    hold = read_csv(HOLD) if HOLD.exists() else pd.DataFrame(columns=HOLD_COLS)
    trades = read_csv(TRADES) if TRADES.exists() else pd.DataFrame(columns=TRADE_COLS)
    hist = read_csv(HIST) if HIST.exists() else pd.DataFrame(columns=HIST_COLS)
    return state, hold, trades, hist


def _save(state, hold, trades, hist):
    PAPER_DIR.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    hold.reindex(columns=HOLD_COLS).to_csv(HOLD, index=False)        # 비어 있어도 열 이름은 남김
    trades.reindex(columns=TRADE_COLS).to_csv(TRADES, index=False)
    hist.reindex(columns=HIST_COLS).to_csv(HIST, index=False)


def _concat(*frames):
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def benchmark_close(market: str, as_of: str):
    """시장 지수의 그날(또는 직전 거래일) 종가. 실패하면 None."""
    try:
        import yfinance as yf
        sym = PAPER_BENCHMARK[market][0]
        h = yf.Ticker(sym).history(start=(pd.Timestamp(as_of) - pd.Timedelta(days=10)).date().isoformat(),
                                   end=(pd.Timestamp(as_of) + pd.Timedelta(days=1)).date().isoformat())
        return float(h["Close"].dropna().iloc[-1])
    except Exception as e:
        print(f"  [모의투자] 지수 받기 실패 ({type(e).__name__})")
        return None


def run(market: str, res: pd.DataFrame, usdkrw: float = 1400.0, bench=benchmark_close) -> dict:
    """res: analyze.analyze() 결과 (그 시장 전체 종목). 반환: 그날 계좌 요약."""
    state, hold, trades, hist = _load()
    as_of = str(res["as_of"].iloc[0])
    acct = state.get(market)
    if acct is None:                                       # 계좌 개설
        start = PAPER_START_KRW if market == "KR" else round(PAPER_START_KRW / usdkrw, 2)
        acct = {"start_date": as_of, "start_cash": start, "cash": start, "start_usdkrw": usdkrw,
                "currency": "KRW" if market == "KR" else "USD", "last_date": None,
                "last_rebalance_week": None, "bench_start": bench(market, as_of)}
        state[market] = acct
        print(f"  [모의투자] {market} 계좌 개설: {start:,.0f} {acct['currency']}")
    if acct["last_date"] and as_of <= acct["last_date"]:
        print(f"  [모의투자] {market} {as_of}는 이미 처리함 → 건너뜀")
        return {}

    cost = PAPER_COSTS[market]
    px = res.set_index("ticker")
    mine = hold[hold["account"] == market].copy()
    others = hold[hold["account"] != market]
    new_trades = []

    def price_of(t):
        return float(px.at[t, "price"]) if t in px.index and pd.notna(px.at[t, "price"]) else None

    def sell(idx, reason):
        h = mine.loc[idx]
        p = price_of(h["ticker"]) * (1 - PAPER_SLIPPAGE)
        gross = h["shares"] * p
        fee = gross * cost["sell"]
        pnl = gross - fee - h["shares"] * h["avg_cost"]
        acct["cash"] += gross - fee
        new_trades.append({"date": as_of, "account": market, "side": "매도", "ticker": h["ticker"], "name": h["name"],
                           "shares": int(h["shares"]), "price": round(p, 4), "amount": round(gross, 2),
                           "cost": round(fee, 2), "pnl": round(pnl, 2),
                           "pnl_pct": round(pnl / (h["shares"] * h["avg_cost"]) * 100, 2), "reason": reason})
        mine.drop(index=idx, inplace=True)

    # 1) 매일 점검: 손절, 비추천 전환
    for idx, h in list(mine.iterrows()):
        p = price_of(h["ticker"])
        if p is None:
            continue                                         # 가격을 못 받은 날은 그대로 둠
        if p / h["avg_cost"] - 1 <= PAPER_STOP_LOSS:
            sell(idx, f"손절 ({(p / h['avg_cost'] - 1) * 100:.1f}%)")
        elif px.at[h["ticker"], "grade"] == "비추천":
            sell(idx, "등급이 비추천으로 떨어짐")

    # 2) 주 1회 교체
    week = pd.Timestamp(as_of).strftime("%G-W%V")
    if acct["last_rebalance_week"] != week:
        rated = res[res["grade"] != "제외"].sort_values("score", ascending=False)
        keep = set(rated["ticker"].head(PAPER_KEEP_RANK))
        for idx, h in list(mine.iterrows()):
            if h["ticker"] not in keep and price_of(h["ticker"]) is not None:
                sell(idx, f"주간 교체: 종합점수 상위 {PAPER_KEEP_RANK}위 밖")

        stock_val = sum(r["shares"] * (price_of(r["ticker"]) or r["avg_cost"]) for _, r in mine.iterrows())
        target = (acct["cash"] + stock_val) / PAPER_HOLDINGS
        per_group = mine["group"].value_counts().to_dict()
        for _, c in rated[rated["grade"] == "추천"].iterrows():
            if len(mine) >= PAPER_HOLDINGS:
                break
            if c["ticker"] in set(mine["ticker"]) or per_group.get(c["group"], 0) >= PAPER_SECTOR_CAP:
                continue
            p = float(c["price"]) * (1 + PAPER_SLIPPAGE)
            shares = math.floor(min(target, acct["cash"]) / (p * (1 + cost["buy"])))
            if shares < 1:
                continue
            gross = shares * p
            fee = gross * cost["buy"]
            acct["cash"] -= gross + fee
            mine.loc[len(hold) + len(mine) + 1000] = {"account": market, "ticker": c["ticker"], "name": c["name"],
                                                      "group": c["group"], "shares": shares,
                                                      "avg_cost": (gross + fee) / shares, "buy_date": as_of}
            per_group[c["group"]] = per_group.get(c["group"], 0) + 1
            new_trades.append({"date": as_of, "account": market, "side": "매수", "ticker": c["ticker"],
                               "name": c["name"], "shares": shares, "price": round(p, 4), "amount": round(gross, 2),
                               "cost": round(fee, 2), "pnl": None, "pnl_pct": None,
                               "reason": f"추천 · 종합 {c['score']:.1f}점 · {c['group']}"})
        acct["last_rebalance_week"] = week

    # 3) 평가와 기록
    stock_val = sum(r["shares"] * (price_of(r["ticker"]) or r["avg_cost"]) for _, r in mine.iterrows())
    nav = acct["cash"] + stock_val
    b = bench(market, as_of)
    if acct.get("bench_start") is None and b is not None:
        acct["bench_start"] = b
    row = {"date": as_of, "account": market, "cash": round(acct["cash"], 2), "stock_value": round(stock_val, 2),
           "nav": round(nav, 2), "return_pct": round((nav / acct["start_cash"] - 1) * 100, 3),
           "bench": b, "bench_return_pct": round((b / acct["bench_start"] - 1) * 100, 3) if b and acct.get("bench_start") else None,
           "holdings": len(mine)}
    acct["last_date"] = as_of
    acct["cash"] = round(acct["cash"], 2)

    hold = _concat(others, mine)
    trades = _concat(trades, pd.DataFrame(new_trades, columns=TRADE_COLS))
    hist = _concat(hist[~((hist["account"] == market) & (hist["date"] == as_of))], pd.DataFrame([row]))
    _save(state, hold, trades, hist)
    print(f"  [모의투자] {market} {as_of}: 매수 {sum(t['side'] == '매수' for t in new_trades)} · "
          f"매도 {sum(t['side'] == '매도' for t in new_trades)} · 보유 {len(mine)} · 수익률 {row['return_pct']:+.2f}%")
    return row


def load_for_dashboard(results: dict, usdkrw: float) -> dict:
    """대시보드용 요약. results: {"KR": res, "US": res}"""
    state, hold, trades, hist = _load()
    out = {}
    for m, acct in state.items():
        res = results.get(m)
        px = res.set_index("ticker") if res is not None else pd.DataFrame()
        h = hold[hold["account"] == m].copy()
        rows = []
        for _, r in h.iterrows():
            info = px.loc[r["ticker"]] if r["ticker"] in px.index else None
            p = float(info["price"]) if info is not None and pd.notna(info["price"]) else r["avg_cost"]
            rows.append({"ticker": r["ticker"], "name": r["name"], "group": r["group"], "shares": int(r["shares"]),
                         "avg_cost": round(float(r["avg_cost"]), 4), "price": round(p, 4), "value": round(p * r["shares"], 2),
                         "pnl_pct": round((p / r["avg_cost"] - 1) * 100, 2), "buy_date": r["buy_date"],
                         "grade": None if info is None else info["grade"],
                         "score": None if info is None else round(float(info["score"]), 1)})
        hh = hist[hist["account"] == m].sort_values("date")
        t = trades[trades["account"] == m].sort_values("date", ascending=False).head(60)
        closed = trades[(trades["account"] == m) & (trades["side"] == "매도")]
        out[m] = {"start_date": acct["start_date"], "start_cash": acct["start_cash"], "currency": acct["currency"],
                  "cash": acct["cash"], "bench_name": PAPER_BENCHMARK[m][1],
                  "history": json.loads(hh[["date", "nav", "return_pct", "bench_return_pct"]].to_json(orient="records")),
                  "holdings": sorted(rows, key=lambda x: -x["value"]),
                  "trades": json.loads(t[["date", "side", "ticker", "name", "shares", "price", "amount", "pnl_pct", "reason"]]
                                       .to_json(orient="records", force_ascii=False)),
                  "closed": int(len(closed)), "wins": int((closed["pnl"] > 0).sum())}
    return out

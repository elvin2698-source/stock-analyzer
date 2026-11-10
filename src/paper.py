"""규칙 기반 모의투자 v2. 분석이 끝날 때마다 시장별·전략별 가상 계좌를 갱신합니다.

하루 처리 순서 (그 시장의 새 거래일 D가 들어왔을 때, 계좌마다)
  1. 기업 이벤트: 지난 처리일 이후의 액면분할(보유 수량·매수가 보정)과 배당(세후 현금 입금)
  2. 대기 주문 체결: 전날까지 낸 주문을 '신호 다음 거래일 시가'에 체결
       - 매도 먼저, 그다음 매수 / 시가에 0.1% 불리하게 + 수수료·세금
       - 한국은 시가가 상한가(+29.5%↑)면 매수 불가, 하한가(-29.5%↓)면 매도 불가(다음 날 재시도)
       - 한 종목 매수 금액은 최근 20일 평균 거래대금의 1%까지
  3. 평가: D 종가로 평가 (가격이 없는 종목은 마지막 가격, 며칠째 없는지 기록)
  4. 새 신호 → 다음 거래일 시가 주문으로 저장
       - 매일: 손절선 도달, 등급 '비추천', 상장폐지·거래정지
       - 교체일(전략별 매주/매달/매일): 상위 N위 밖 매도, 하락장 방어(지수 200일선 아래면 절반만 보유),
         빈자리는 '추천' 종목 중 점수 순(대조군은 무작위)으로 매수 주문
파일 (paper/): state.json, holdings.csv, orders.csv, trades.csv, history.csv, bench.csv, ew.csv
"""
import json
import math
import random
import warnings

import numpy as np
import pandas as pd

from config import (PAPER_DIR, PAPER_START_KRW, PAPER_SLIPPAGE, PAPER_MAX_ADV, PAPER_LIMIT_KR, PAPER_FX_COST,
                    PAPER_COSTS, PAPER_DIV_TAX, PAPER_US_CGT, PAPER_RF, PAPER_REGIME_INDEX, PAPER_BENCHMARKS,
                    PAPER_STRATEGIES, PAPER_RULES_VERSION, read_csv)

warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA")

STATE = PAPER_DIR / "state.json"
HOLD = PAPER_DIR / "holdings.csv"
ORDERS = PAPER_DIR / "orders.csv"
TRADES = PAPER_DIR / "trades.csv"
HIST = PAPER_DIR / "history.csv"
BENCH = PAPER_DIR / "bench.csv"
EW = PAPER_DIR / "ew.csv"

HOLD_COLS = ["account", "ticker", "name", "group", "shares", "avg_cost", "buy_date", "last_price", "last_price_date"]
ORDER_COLS = ["account", "signal_date", "side", "ticker", "name", "group", "score", "rank", "target_n", "reason"]
TRADE_COLS = ["date", "account", "side", "ticker", "name", "shares", "price", "amount", "cost", "pnl", "pnl_pct",
              "held_days", "reason", "rules"]
HIST_COLS = ["date", "account", "cash", "stock_value", "nav", "return_pct", "holdings", "usdkrw"]


# ───────────────────────── 저장·불러오기 ─────────────────────────
def _empty(cols):
    return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})


def _read(path, cols):
    return read_csv(path).reindex(columns=cols) if path.exists() else _empty(cols)


def load():
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    files = {"hold": _read(HOLD, HOLD_COLS), "orders": _read(ORDERS, ORDER_COLS),
             "trades": _read(TRADES, TRADE_COLS), "hist": _read(HIST, HIST_COLS)}
    if state and state.get("version") != 2:
        state, files = migrate_v1(state, files)
    state.setdefault("version", 2)
    state.setdefault("accounts", {})
    return state, files


def save(state, files):
    PAPER_DIR.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, path, cols in (("hold", HOLD, HOLD_COLS), ("orders", ORDERS, ORDER_COLS),
                            ("trades", TRADES, TRADE_COLS), ("hist", HIST, HIST_COLS)):
        files[key].reindex(columns=cols).to_csv(path, index=False)


def migrate_v1(state, files):
    """v1(시장당 균형형 1계좌, 같은 날 종가 체결) 기록을 지우지 않고 v2 형식으로 옮깁니다."""
    accounts = {}
    for m, a in state.items():
        acc = f"{m}-balanced"
        accounts[acc] = {"market": m, "strategy": "balanced", "name": PAPER_STRATEGIES["balanced"]["name"],
                         "start_date": a["start_date"], "start_cash": a["start_cash"], "currency": a["currency"],
                         "cash": a["cash"], "start_usdkrw": a.get("start_usdkrw"), "last_date": a["last_date"],
                         "last_rebalance": a.get("last_rebalance_week"), "rules_since": {"v1": a["start_date"]}}
        for key in ("hold", "trades", "hist"):
            f = files[key]
            files[key] = f.assign(account=f["account"].replace({m: acc}))
        if m == "US":   # v1에서 빠졌던 환전 수수료를 한 번 소급 반영
            fee = round(a["start_cash"] * PAPER_FX_COST, 2)
            accounts[acc]["cash"] = round(accounts[acc]["cash"] - fee, 2)
            files["trades"] = _concat(files["trades"], pd.DataFrame([{
                "date": a["last_date"], "account": acc, "side": "비용", "ticker": "", "name": "환전 수수료",
                "shares": 0, "price": 0, "amount": fee, "cost": fee, "reason": "v2 전환: 시작 때 빠진 환전 수수료 0.25% 소급 반영",
                "rules": PAPER_RULES_VERSION}]))
    files["trades"]["rules"] = files["trades"]["rules"].fillna("v1")
    h = files["hold"]
    files["hold"] = h.assign(last_price=h["last_price"].fillna(h["avg_cost"]),
                             last_price_date=h["last_price_date"].fillna(h["buy_date"]))
    print("  [모의투자] v1 기록을 v2 형식으로 옮김 (기존 매매·수익률 유지)")
    return {"version": 2, "accounts": accounts}, files


def _concat(*frames, cols=None):
    """빈 표는 빼고 이어 붙임. cols를 주면 결과가 비어 있어도 열 이름을 유지."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return _empty(cols) if cols is not None else pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        out = pd.concat(frames, ignore_index=True)
    return out.reindex(columns=cols) if cols is not None else out


# ───────────────────────── 가격 도우미 ─────────────────────────
class Prices:
    """한 시장의 저장된 주가(액면분할 반영 가격)를 날짜×종목 표로 들고 있는 도우미."""

    def __init__(self, prices: pd.DataFrame):
        p = prices.drop_duplicates(["date", "ticker"], keep="last")
        self.close = p.pivot(index="date", columns="ticker", values="close").sort_index()
        self.open = p.pivot(index="date", columns="ticker", values="open").reindex_like(self.close) \
            if "open" in p else self.close
        self.volume = p.pivot(index="date", columns="ticker", values="volume").reindex_like(self.close)
        ev = p[(p.get("dividend", 0) > 0) | (p.get("split", 0) > 0)] if "dividend" in p else p.iloc[0:0]
        self.events = ev[["date", "ticker", "dividend", "split"]] if not ev.empty else \
            pd.DataFrame(columns=["date", "ticker", "dividend", "split"])
        self.dates = list(self.close.index)

    def has(self, t):
        return t in self.close.columns

    def close_on(self, t, d):
        """d 이하에서 가장 최근 종가와 그 날짜."""
        if not self.has(t):
            return None, None
        s = self.close[t].loc[:d].dropna()
        return (float(s.iloc[-1]), s.index[-1]) if len(s) else (None, None)

    def next_day(self, t, after, upto):
        """after 다음 거래일(≤ upto) 중 그 종목 가격이 있는 첫날: (날짜, 시가, 전날 종가)."""
        if not self.has(t):
            return None
        c = self.close[t]
        for d in self.dates:
            if d <= after or d > upto:
                continue
            if pd.notna(c.get(d)):
                o = self.open[t].get(d)
                o = float(o) if pd.notna(o) and o > 0 else float(c[d])
                prev = c.loc[:d].dropna()
                prev = float(prev.iloc[-2]) if len(prev) >= 2 else None
                return d, o, prev
        return None

    def adv(self, t, upto):
        """upto까지 최근 20일 평균 거래대금."""
        if not self.has(t):
            return 0.0
        v = (self.close[t] * self.volume[t]).loc[:upto].dropna().tail(20)
        return float(v.mean()) if len(v) else 0.0


# ───────────────────────── 지수·동일가중 지수 ─────────────────────────
def fetch_bench(market):
    """시장 지수 2년치 종가. 실패한 지수는 건너뜀."""
    import yfinance as yf
    rows = []
    for sym, _ in PAPER_BENCHMARKS[market]:
        try:
            h = yf.Ticker(sym).history(period="2y", auto_adjust=False)["Close"].dropna()
            rows.append(pd.DataFrame({"date": h.index.date.astype(str), "symbol": sym, "close": h.values}))
        except Exception as e:
            print(f"  [모의투자] 지수 {sym} 받기 실패 ({type(e).__name__})")
    return _concat(*rows)


def update_bench(market, fetch=fetch_bench):
    old = read_csv(BENCH) if BENCH.exists() else pd.DataFrame(columns=["date", "symbol", "close"])
    new = fetch(market)
    if new is None or new.empty:
        return old
    allb = _concat(old, new).drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"])
    allb.to_csv(BENCH, index=False)
    return allb


def regime_is_bear(market, bench, d):
    s = bench[bench["symbol"] == PAPER_REGIME_INDEX[market]].set_index("date")["close"].sort_index().loc[:d]
    if len(s) < 200:
        return False
    return bool(s.iloc[-1] < s.tail(200).mean())


def update_ew(market, px: Prices, universe, since):
    """동일가중 지수: 분석 대상(제외 아닌) 전 종목을 매일 똑같은 금액으로 들고 있었다면 (배당 포함).
    since 부터 지금까지 빠진 날만 채워 넣습니다. 하루 수익률은 ±50%로 잘라 데이터 오류를 막습니다."""
    old = read_csv(EW) if EW.exists() else pd.DataFrame(columns=["date", "market", "ret", "level"])
    mine = old[old["market"] == market].sort_values("date")
    last = mine["date"].max() if len(mine) else since
    level = float(mine["level"].iloc[-1]) if len(mine) else 100.0
    cols = [t for t in universe if px.has(t)]
    if not cols:
        return old
    c = px.close[cols]
    div = pd.DataFrame(0.0, index=c.index, columns=cols)
    ev = px.events[px.events["ticker"].isin(cols) & (px.events["dividend"] > 0)]
    for d, t, v in ev[["date", "ticker", "dividend"]].itertuples(index=False):
        if d in div.index:
            div.at[d, t] = v
    rets = ((c + div) / c.ffill().shift(1) - 1).clip(-0.5, 0.5)
    rows = []
    if not len(mine):
        rows.append({"date": since, "market": market, "ret": 0.0, "level": 100.0})
    for d in [x for x in px.dates if x > last]:
        r = rets.loc[d].dropna()
        if len(r) < 10:
            continue
        level *= 1 + float(r.mean())
        rows.append({"date": d, "market": market, "ret": round(float(r.mean()), 6), "level": round(level, 4)})
    out = _concat(old, pd.DataFrame(rows))
    out.to_csv(EW, index=False)
    return out


# ───────────────────────── 계좌 처리 ─────────────────────────
def ensure_accounts(state, market, d, usdkrw):
    for key, st in PAPER_STRATEGIES.items():
        acc = f"{market}-{key}"
        if acc in state["accounts"]:
            continue
        start = PAPER_START_KRW if market == "KR" else round(PAPER_START_KRW / (usdkrw * (1 + PAPER_FX_COST)), 2)
        state["accounts"][acc] = {"market": market, "strategy": key, "name": st["name"], "start_date": d,
                                  "start_cash": start, "currency": "KRW" if market == "KR" else "USD", "cash": start,
                                  "start_usdkrw": usdkrw, "last_date": None, "last_rebalance": None,
                                  "rules_since": {PAPER_RULES_VERSION: d}}
        print(f"  [모의투자] {acc} 계좌 개설 ({st['name']}): {start:,.0f} {'원' if market == 'KR' else '달러'}")


def _rebalance_key(rule, d):
    t = pd.Timestamp(d)
    return {"weekly": t.strftime("%G-W%V"), "monthly": t.strftime("%Y-%m"), "daily": d}[rule]


def process_account(acc, a, market, d, res, px, bench, files, log):
    st = PAPER_STRATEGIES[a["strategy"]]
    cost = PAPER_COSTS[market]
    hold = files["hold"]
    mine = hold[hold["account"] == acc].copy().reset_index(drop=True)
    trades = []

    def trade(**k):
        row = {"date": d, "account": acc, "rules": PAPER_RULES_VERSION, "pnl": None, "pnl_pct": None, "held_days": None}
        row.update(k)
        trades.append(row)

    since = a["last_date"] or a["start_date"]
    # 1) 기업 이벤트 (액면분할 → 배당 순)
    if a["last_date"]:
        ev = px.events[(px.events["date"] > since) & (px.events["date"] <= d) & px.events["ticker"].isin(mine["ticker"])]
        for ed, t, dv, sp in ev.sort_values("date")[["date", "ticker", "dividend", "split"]].itertuples(index=False):
            i = mine.index[mine["ticker"] == t][0]
            if sp and sp > 0 and sp != 1:
                new_sh = mine.at[i, "shares"] * sp
                whole = math.floor(new_sh + 1e-9)
                frac = new_sh - whole
                p_now, _ = px.close_on(t, ed)
                if frac > 1e-9 and p_now:
                    a["cash"] += frac * p_now
                mine.at[i, "avg_cost"] = mine.at[i, "avg_cost"] * mine.at[i, "shares"] / max(whole, 1)
                mine.at[i, "shares"] = whole
                mine.at[i, "last_price"] = mine.at[i, "last_price"] / sp
                trade(date=ed, side="분할", ticker=t, name=mine.at[i, "name"], shares=whole, price=0, amount=0, cost=0,
                      reason=f"액면분할 {sp:g}:1 → 보유 수량·매수가 보정" + (" (1주 미만 현금 정산)" if frac > 1e-9 else ""))
            if dv and dv > 0:
                net = mine.at[i, "shares"] * dv * (1 - PAPER_DIV_TAX[market])
                a["cash"] += net
                trade(date=ed, side="배당", ticker=t, name=mine.at[i, "name"], shares=int(mine.at[i, "shares"]),
                      price=dv, amount=round(net, 2), cost=round(mine.at[i, "shares"] * dv * PAPER_DIV_TAX[market], 2),
                      reason=f"주당 {dv:g} 배당 (세금 {PAPER_DIV_TAX[market] * 100:.1f}% 뗀 금액 입금)")
        mine = mine[mine["shares"] > 0].reset_index(drop=True)

    # 2) 대기 주문 체결
    orders = files["orders"]
    mine_orders = orders[(orders["account"] == acc) & (orders["signal_date"] < d)]
    keep_orders = []
    for _, o in mine_orders[mine_orders["side"] == "매도"].iterrows():
        idx = mine.index[mine["ticker"] == o["ticker"]]
        if not len(idx):
            continue
        i = idx[0]
        nd = px.next_day(o["ticker"], o["signal_date"], d)
        if nd is None:
            keep_orders.append(o)                                   # 가격이 아직 없음(거래정지 등) → 다음에 재시도
            continue
        ed, op, prev = nd
        if market == "KR" and prev and op / prev - 1 <= -PAPER_LIMIT_KR:
            keep_orders.append(o)
            log.append(f"{acc} {o['name']} 하한가로 매도 못 함 → 다음 날 재시도")
            continue
        h = mine.loc[i]
        p = op * (1 - PAPER_SLIPPAGE)
        gross = h["shares"] * p
        fee = gross * cost["sell"]
        basis = h["shares"] * h["avg_cost"]
        a["cash"] += gross - fee
        trade(date=ed, side="매도", ticker=h["ticker"], name=h["name"], shares=int(h["shares"]), price=round(p, 4),
              amount=round(gross, 2), cost=round(fee, 2), pnl=round(gross - fee - basis, 2),
              pnl_pct=round((gross - fee) / basis * 100 - 100, 2),
              held_days=(pd.Timestamp(ed) - pd.Timestamp(h["buy_date"])).days, reason=o["reason"])
        mine = mine.drop(index=i).reset_index(drop=True)

    buys = mine_orders[mine_orders["side"] == "매수"].sort_values("rank")
    if len(buys):
        target_n = int(buys["target_n"].iloc[0])
        nav_now = a["cash"] + sum(r["shares"] * (px.close_on(r["ticker"], d)[0] or r["last_price"]) for _, r in mine.iterrows())
        target_amt = nav_now / st["n"]
        per_group = mine["group"].value_counts().to_dict()
        for _, o in buys.iterrows():
            if len(mine) >= target_n:
                break
            if o["ticker"] in set(mine["ticker"]) or per_group.get(o["group"], 0) >= st["sector_cap"]:
                continue
            nd = px.next_day(o["ticker"], o["signal_date"], d)
            if nd is None:
                continue
            ed, op, prev = nd
            if market == "KR" and prev and op / prev - 1 >= PAPER_LIMIT_KR:
                log.append(f"{acc} {o['name']} 상한가로 매수 못 함")
                continue
            p = op * (1 + PAPER_SLIPPAGE)
            cap = px.adv(o["ticker"], o["signal_date"]) * PAPER_MAX_ADV
            amt = min(target_amt, a["cash"], cap)
            shares = math.floor(amt / (p * (1 + cost["buy"])))
            if shares < 1:
                continue
            gross = shares * p
            fee = gross * cost["buy"]
            a["cash"] -= gross + fee
            mine.loc[len(mine)] = {"account": acc, "ticker": o["ticker"], "name": o["name"], "group": o["group"],
                                   "shares": shares, "avg_cost": (gross + fee) / shares, "buy_date": ed,
                                   "last_price": op, "last_price_date": ed}
            per_group[o["group"]] = per_group.get(o["group"], 0) + 1
            note = " · 거래대금 한도로 일부만" if cap < min(target_amt, a["cash"] + gross + fee) else ""
            trade(date=ed, side="매수", ticker=o["ticker"], name=o["name"], shares=shares, price=round(p, 4),
                  amount=round(gross, 2), cost=round(fee, 2), reason=o["reason"] + note)
    files["orders"] = _concat(orders[~orders.index.isin(mine_orders.index)], pd.DataFrame(keep_orders), cols=ORDER_COLS)

    # 3) 평가
    stale = 0
    for i, r in mine.iterrows():
        p, pd_ = px.close_on(r["ticker"], d)
        if p is not None:
            mine.at[i, "last_price"], mine.at[i, "last_price_date"] = p, pd_
        if mine.at[i, "last_price_date"] < d:
            stale += 1
    stock_val = float((mine["shares"] * mine["last_price"]).sum()) if len(mine) else 0.0
    nav = a["cash"] + stock_val
    if stale:
        log.append(f"{acc}: {stale}종목은 오늘 가격이 없어 마지막 가격으로 평가")

    # 4) 새 신호 (다음 거래일 시가 주문)
    new_orders = []
    pending_sell = set(files["orders"].loc[(files["orders"]["account"] == acc) & (files["orders"]["side"] == "매도"), "ticker"])
    r_idx = res.set_index("ticker")

    def order(side, row, reason, score=None, rank=0, target_n=0):
        new_orders.append({"account": acc, "signal_date": d, "side": side, "ticker": row["ticker"], "name": row["name"],
                           "group": row["group"], "score": score, "rank": rank, "target_n": target_n, "reason": reason})

    selling = set(pending_sell)
    for _, h in mine.iterrows():
        t = h["ticker"]
        if t in selling:
            continue
        chg = h["last_price"] / h["avg_cost"] - 1
        info = r_idx.loc[t] if t in r_idx.index else None
        why = None
        if chg <= st["stop"]:
            why = f"손절 ({chg * 100:.1f}%, 기준 {st['stop'] * 100:.0f}%)"
        elif info is None:
            why = "분석 대상에서 빠짐 (상장폐지·데이터 없음)"
        elif info["grade"] == "비추천":
            why = f"등급이 비추천으로 떨어짐 ({info['score']:.1f}점)"
        elif str(info.get("instrument", "")) == "거래정지":
            why = "거래정지"
        if why:
            order("매도", h, why)
            selling.add(t)

    key = _rebalance_key(st["rebalance"], d)
    if a.get("last_rebalance") != key:
        elig = res[res["grade"] != "제외"]
        if st["sizes"]:
            elig = elig[elig["size"].isin(st["sizes"])]
        ranked = elig.sort_values("score", ascending=False)
        keep = set(ranked["ticker"].head(st["keep_rank"]))
        for _, h in mine.iterrows():
            if h["ticker"] not in selling and h["ticker"] not in keep:
                order("매도", h, f"교체: 점수 상위 {st['keep_rank']}위 밖")
                selling.add(h["ticker"])
        target_n = st["n"]
        if st["regime"] and regime_is_bear(market, bench, d):
            target_n = st["n"] // 2
            remain = mine[~mine["ticker"].isin(selling)].copy()
            remain["score"] = remain["ticker"].map(r_idx["score"]).fillna(0)
            for _, h in remain.sort_values("score").head(max(0, len(remain) - target_n)).iterrows():
                order("매도", h, f"하락장 방어: 지수가 200일선 아래 → {target_n}종목으로 축소")
                selling.add(h["ticker"])
            log.append(f"{acc}: 하락장 판단 → 목표 {target_n}종목")
        cand = ranked[(ranked["grade"] == "추천") & ~ranked["ticker"].isin(set(mine["ticker"]))]
        if st["pick"] == "random":
            cand = cand.sample(frac=1, random_state=random.Random(f"{acc}{d}").randrange(1 << 30))
        for rank, (_, c) in enumerate(cand.head(st["n"] * 3).iterrows(), 1):
            why = f"추천 · 종합 {c['score']:.1f}점 · {c['group']}" + (" · 무작위 선택" if st["pick"] == "random" else "")
            order("매수", c, why, score=c["score"], rank=rank, target_n=target_n)
        a["last_rebalance"] = key

    files["orders"] = _concat(files["orders"], pd.DataFrame(new_orders, columns=ORDER_COLS), cols=ORDER_COLS)
    files["hold"] = _concat(hold[hold["account"] != acc], mine, cols=HOLD_COLS)
    files["trades"] = _concat(files["trades"], pd.DataFrame(trades, columns=TRADE_COLS), cols=TRADE_COLS)
    files["hist"] = _concat(files["hist"][~((files["hist"]["account"] == acc) & (files["hist"]["date"] == d))],
                            pd.DataFrame([{"date": d, "account": acc, "cash": round(a["cash"], 2),
                                           "stock_value": round(stock_val, 2), "nav": round(nav, 2),
                                           "return_pct": round((nav / a["start_cash"] - 1) * 100, 3),
                                           "holdings": len(mine), "usdkrw": a.get("usdkrw_now")}]), cols=HIST_COLS)
    a["cash"] = round(a["cash"], 2)
    a["last_date"] = d
    n_b = sum(t["side"] == "매수" for t in trades)
    n_s = sum(t["side"] == "매도" for t in trades)
    print(f"  [모의투자] {acc} {d}: 매수 {n_b} · 매도 {n_s} · 보유 {len(mine)} · 다음 주문 {len(new_orders)} · "
          f"수익률 {(nav / a['start_cash'] - 1) * 100:+.2f}%")


def run(market: str, res: pd.DataFrame, prices: pd.DataFrame, usdkrw: float = 1400.0, fetch=fetch_bench):
    """res: analyze.analyze() 결과(그 시장 전 종목), prices: 그 시장 저장 주가."""
    state, files = load()
    d = str(res["as_of"].iloc[0])
    px = Prices(prices)
    bench = update_bench(market, fetch)
    ensure_accounts(state, market, d, usdkrw)
    accts = {k: v for k, v in state["accounts"].items() if v["market"] == market}
    since = min(v["start_date"] for v in accts.values())
    update_ew(market, px, res.loc[res["grade"] != "제외", "ticker"].tolist(), since)
    log = []
    for acc, a in sorted(accts.items()):
        if a["last_date"] and d <= a["last_date"]:
            continue
        a["usdkrw_now"] = usdkrw
        process_account(acc, a, market, d, res, px, bench, files, log)
    for line in log:
        print("  [모의투자] " + line)
    save(state, files)
    return state


# ───────────────────────── 성과 지표 ─────────────────────────
def _series_return(s: pd.Series, start, end=None):
    s = s.dropna().sort_index()
    if end is not None:
        s = s.loc[:end]
    s0 = s.loc[:start]
    base = s0.iloc[-1] if len(s0) else (s.iloc[0] if len(s) else None)
    return None if base is None or not len(s) else float(s.iloc[-1] / base - 1) * 100


def metrics(nav: pd.Series, index_close: pd.Series = None):
    """nav: 날짜별 평가금액. 수익률 %, 최대낙폭 %, 연변동성 %, 샤프, 베타."""
    nav = nav.dropna().sort_index()
    out = {"days": int(len(nav))}
    if len(nav) < 2:
        return out
    r = nav.pct_change().dropna()
    dd = nav / nav.cummax() - 1
    out["mdd"] = round(float(dd.min()) * 100, 2)
    if len(r) >= 20:
        vol = float(r.std()) * np.sqrt(252)
        out["vol"] = round(vol * 100, 2)
        out["sharpe"] = round((float(r.mean()) * 252 - PAPER_RF) / vol, 2) if vol > 0 else None
        if index_close is not None:
            ir = index_close.sort_index().pct_change().reindex(r.index).dropna()
            rr = r.reindex(ir.index)
            if len(ir) >= 20 and ir.var() > 0:
                out["beta"] = round(float(np.cov(rr, ir)[0, 1] / ir.var()), 2)
    return out


def load_for_dashboard(results: dict, usdkrw: float) -> dict:
    """대시보드용 요약. results: {"KR": res, "US": res}"""
    if not STATE.exists():
        return {}
    state, files = load()
    bench = read_csv(BENCH) if BENCH.exists() else pd.DataFrame(columns=["date", "symbol", "close"])
    ew = read_csv(EW) if EW.exists() else pd.DataFrame(columns=["date", "market", "ret", "level"])
    hold, orders, trades, hist = files["hold"], files["orders"], files["trades"], files["hist"]
    out = {"rules": PAPER_RULES_VERSION, "strategies": {k: {kk: v[kk] for kk in ("name", "n", "sector_cap", "stop",
           "keep_rank", "rebalance", "sizes", "pick", "regime")} for k, v in PAPER_STRATEGIES.items()},
           "costs": PAPER_COSTS, "markets": {}}
    year = pd.Timestamp.today().strftime("%Y")
    for m in ("KR", "US"):
        accts = {k: v for k, v in state["accounts"].items() if v["market"] == m}
        if not accts:
            continue
        res = results.get(m)
        r_idx = res.set_index("ticker") if res is not None else pd.DataFrame()
        common = max(v["start_date"] for v in accts.values())
        bser = {sym: bench[bench["symbol"] == sym].set_index("date")["close"].astype(float) for sym, _ in PAPER_BENCHMARKS[m]}
        ews = ew[ew["market"] == m].set_index("date")["level"].astype(float)
        main_idx = bser.get(PAPER_REGIME_INDEX[m])
        earliest = min(v["start_date"] for v in accts.values())
        dates = sorted(set(hist.loc[hist["account"].isin(accts), "date"]) | {earliest})
        curves = {"dates": dates, "series": {}}
        for sym, nm in PAPER_BENCHMARKS[m]:
            s = bser[sym].reindex(sorted(set(bser[sym].index) | set(dates))).ffill().reindex(dates)
            base = bser[sym].loc[:earliest]
            if len(base) and len(s.dropna()):
                curves["series"][nm] = [None if pd.isna(v) else round((v / base.iloc[-1] - 1) * 100, 3) for v in s]
        if len(ews):
            s = ews.reindex(sorted(set(ews.index) | set(dates))).ffill().reindex(dates)
            b0 = ews.loc[:earliest]
            if len(b0):
                curves["series"]["동일가중(전 종목)"] = [None if pd.isna(v) else round((v / b0.iloc[-1] - 1) * 100, 3) for v in s]
        accounts = []
        for acc, a in sorted(accts.items(), key=lambda kv: list(PAPER_STRATEGIES).index(kv[1]["strategy"])):
            h = hist[hist["account"] == acc].sort_values("date")
            navs = h.set_index("date")["nav"].astype(float)
            last_nav = float(navs.iloc[-1]) if len(navs) else a["start_cash"]
            ret = (last_nav / a["start_cash"] - 1) * 100
            mx = metrics(pd.concat([pd.Series({a["start_date"]: a["start_cash"]}), navs]).groupby(level=0).last(), main_idx)
            t = trades[trades["account"] == acc]
            sells = t[t["side"] == "매도"]
            buys_sells = t[t["side"].isin(["매수", "매도"])]
            avg_nav = float(navs.mean()) if len(navs) else a["start_cash"]
            # 공통 기간(모든 전략이 함께 운용된 날부터) 수익률
            base_c = navs.loc[:common]
            ret_common = (last_nav / float(base_c.iloc[-1]) - 1) * 100 if len(base_c) else (0.0 if a["start_date"] == common else None)
            end = a["last_date"] or a["start_date"]
            bench_ret = {nm: _series_return(bser[sym], a["start_date"], end) for sym, nm in PAPER_BENCHMARKS[m]}
            ew_ret = _series_return(ews, a["start_date"], end) if len(ews) else None
            hold_rows = []
            for _, r in hold[hold["account"] == acc].iterrows():
                info = r_idx.loc[r["ticker"]] if r["ticker"] in r_idx.index else None
                hold_rows.append({"ticker": r["ticker"], "name": r["name"], "group": r["group"], "shares": int(r["shares"]),
                                  "avg_cost": round(float(r["avg_cost"]), 4), "price": round(float(r["last_price"]), 4),
                                  "value": round(float(r["last_price"]) * r["shares"], 2),
                                  "pnl_pct": round((float(r["last_price"]) / float(r["avg_cost"]) - 1) * 100, 2),
                                  "buy_date": r["buy_date"], "stale": r["last_price_date"] < a["last_date"] if a["last_date"] else False,
                                  "grade": None if info is None else info["grade"],
                                  "score": None if info is None else round(float(info["score"]), 1)})
            o = orders[orders["account"] == acc]
            pend = [{"side": x["side"], "name": x["name"], "ticker": x["ticker"], "reason": x["reason"], "rank": int(x["rank"])}
                    for _, x in o.sort_values(["side", "rank"]).iterrows()]
            n_buy_target = int(o.loc[o["side"] == "매수", "target_n"].max()) if (o["side"] == "매수").any() else 0
            item = {"id": acc, "strategy": a["strategy"], "name": a["name"], "start_date": a["start_date"],
                    "start_cash": a["start_cash"], "currency": a["currency"], "cash": a["cash"], "nav": round(last_nav, 2),
                    "return_pct": round(ret, 2), "return_common": None if ret_common is None else round(ret_common, 2),
                    "bench": bench_ret, "ew": ew_ret, "metrics": mx,
                    "trades_n": int(len(buys_sells)), "closed": int(len(sells)), "wins": int((sells["pnl"].astype(float) > 0).sum()),
                    "avg_win": round(float(sells.loc[sells["pnl"].astype(float) > 0, "pnl_pct"].astype(float).mean()), 2) if (sells["pnl"].astype(float) > 0).any() else None,
                    "avg_loss": round(float(sells.loc[sells["pnl"].astype(float) <= 0, "pnl_pct"].astype(float).mean()), 2) if (sells["pnl"].astype(float) <= 0).any() else None,
                    "avg_hold": round(float(sells["held_days"].astype(float).mean()), 1) if sells["held_days"].notna().any() else None,
                    "turnover": round(float(buys_sells["amount"].astype(float).sum()) / 2 / avg_nav * 100, 1) if avg_nav else None,
                    "dividends": round(float(t.loc[t["side"] == "배당", "amount"].astype(float).sum()), 2),
                    "costs": round(float(t["cost"].astype(float).sum()), 2),
                    "holdings": sorted(hold_rows, key=lambda x: -x["value"]), "pending": pend, "pending_target": n_buy_target,
                    "curve": [None if d not in navs.index else round((float(navs[d]) / a["start_cash"] - 1) * 100, 3) for d in dates],
                    "trades": json.loads(t.sort_values("date", ascending=False).head(80)
                                         [["date", "side", "ticker", "name", "shares", "price", "amount", "pnl_pct", "reason", "rules"]]
                                         .to_json(orient="records", force_ascii=False))}
            if a["start_date"] == earliest:
                item["curve"][0] = 0.0 if item["curve"][0] is None else item["curve"][0]
            if m == "US":
                gains = float(sells.loc[sells["date"].astype(str).str[:4] == year, "pnl"].astype(float).sum()) * usdkrw
                rate, ded = PAPER_US_CGT
                item["krw_value"] = round(last_nav * usdkrw)
                item["krw_return"] = round((last_nav * usdkrw / PAPER_START_KRW - 1) * 100, 2)
                item["tax_est_krw"] = round(max(0.0, gains - ded) * rate)
            accounts.append(item)
        out["markets"][m] = {"accounts": accounts, "curves": curves, "common_start": common,
                             "bench_names": [nm for _, nm in PAPER_BENCHMARKS[m]]}
    return out


# ───────────────────────── 주간 리포트 ─────────────────────────
def weekly_report(today: str = None) -> dict:
    """이번 주(월~오늘) 계좌별 성과와 매매를 paper/reports/<연도-주>.md 로 저장하고 요약을 돌려줍니다."""
    if not STATE.exists():
        return {}
    state, files = load()
    hist, trades = files["hist"], files["trades"]
    if hist.empty:
        return {}
    today = today or str(hist["date"].max())
    week = pd.Timestamp(today).strftime("%G-W%V")
    monday = (pd.Timestamp(today) - pd.Timedelta(days=pd.Timestamp(today).weekday())).date().isoformat()
    lines = [f"# 모의투자 주간 리포트 {week}", "", f"기간: {monday} ~ {today} · 규칙 {PAPER_RULES_VERSION}", ""]
    summary = {"week": week, "from": monday, "to": today, "accounts": []}
    for m, label in (("KR", "한국"), ("US", "미국")):
        accts = [(k, v) for k, v in state["accounts"].items() if v["market"] == m]
        if not accts:
            continue
        lines += [f"## {label}", "", "| 계좌 | 이번 주 | 시작 이후 | 보유 | 이번 주 매수·매도 |", "|---|---:|---:|---:|---:|"]
        for acc, a in sorted(accts, key=lambda kv: list(PAPER_STRATEGIES).index(kv[1]["strategy"])):
            h = hist[hist["account"] == acc].sort_values("date")
            if h.empty:
                continue
            before = h[h["date"] < monday]
            base = float(before["nav"].iloc[-1]) if len(before) else a["start_cash"]
            now = float(h["nav"].iloc[-1])
            t = trades[(trades["account"] == acc) & (trades["date"] >= monday)]
            nb, ns = int((t["side"] == "매수").sum()), int((t["side"] == "매도").sum())
            wk = (now / base - 1) * 100
            tot = (now / a["start_cash"] - 1) * 100
            lines.append(f"| {a['name']} | {wk:+.2f}% | {tot:+.2f}% | {int(h['holdings'].iloc[-1])} | {nb} · {ns} |")
            summary["accounts"].append({"id": acc, "name": a["name"], "market": m, "week": round(wk, 2),
                                        "total": round(tot, 2), "buys": nb, "sells": ns})
        lines.append("")
        t = trades[trades["account"].isin([k for k, _ in accts]) & (trades["date"] >= monday) & (trades["side"] == "매도")]
        if len(t):
            lines += ["**이번 주 매도**", ""] + [f"- {r['date']} {r['name']} {float(r['pnl_pct']):+.1f}% · {r['reason']}"
                                             for _, r in t.sort_values("date").iterrows()] + [""]
    out = PAPER_DIR / "reports" / f"{week}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return summary

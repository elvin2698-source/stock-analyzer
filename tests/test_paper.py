"""모의투자 엔진 검사: 체결 시점, 비용, 배당, 액면분할, 상한가, 거래대금 한도, 현금 회계."""
import pandas as pd
import pytest

import analyze
import paper
from config import PAPER_STRATEGIES, PAPER_DIV_TAX
from conftest import make_market, flat_bench


def run_days(comp, prices, dates, start_i, end_i, market="KR"):
    for i in range(start_i, end_i):
        d = dates[i]
        p = prices[prices["date"] <= d]
        res = analyze.analyze(p, comp, market)
        res["grade"] = res["grade"].replace({"중립": "추천"}) if (res["grade"] == "추천").sum() < 40 else res["grade"]
        paper.run(market, res, p, 1400.0, fetch=flat_bench)


def cash_from_trades(acc, a):
    t = pd.read_csv(paper.TRADES)
    t = t[t["account"] == acc]
    buy = t[t["side"] == "매수"]
    sell = t[t["side"] == "매도"]
    div = t[t["side"] == "배당"]
    other = t[t["side"] == "비용"]
    return a["start_cash"] - (buy["amount"] + buy["cost"]).sum() + (sell["amount"] - sell["cost"]).sum() \
        + div["amount"].sum() - other["amount"].sum()


def test_accounts_trade_next_day_open_and_cash_matches(paper_dir):
    comp, prices, dates = make_market()
    run_days(comp, prices, dates, 230, 250)
    state, files = paper.load()
    assert set(state["accounts"]) == {f"KR-{k}" for k in PAPER_STRATEGIES}
    t = files["trades"]
    buys = t[t["side"] == "매수"]
    assert len(buys) > 0
    # 첫 매수는 계좌 개설 다음 거래일 (신호일 종가가 아니라 다음날 시가)
    assert buys["date"].min() == dates[231]
    o = prices.set_index(["date", "ticker"])["open"]
    for _, b in buys.head(20).iterrows():
        assert b["price"] == pytest.approx(o[(b["date"], b["ticker"])] * 1.001, rel=1e-6)
    for acc, a in state["accounts"].items():
        assert a["cash"] >= -1e-6
        assert cash_from_trades(acc, a) == pytest.approx(a["cash"], abs=1.0)
        h = files["hold"][files["hold"]["account"] == acc]
        n = PAPER_STRATEGIES[a["strategy"]]["n"]
        assert len(h) <= n
        assert h["group"].value_counts().max() <= PAPER_STRATEGIES[a["strategy"]]["sector_cap"]


def test_same_day_is_not_processed_twice(paper_dir):
    comp, prices, dates = make_market()
    run_days(comp, prices, dates, 230, 235)
    before = pd.read_csv(paper.TRADES)
    run_days(comp, prices, dates, 234, 235)
    after = pd.read_csv(paper.TRADES)
    assert len(before) == len(after)


def test_split_does_not_trigger_stop_loss_and_dividend_is_paid(paper_dir):
    comp, prices, dates = make_market()
    run_days(comp, prices, dates, 230, 240)
    state, files = paper.load()
    acc = "KR-balanced"
    held = files["hold"][files["hold"]["account"] == acc].iloc[0]
    t = held["ticker"]
    shares0 = held["shares"]
    # 다음 날 2:1 분할 (Yahoo처럼 이전 가격은 이미 반으로 보정돼 있다고 가정) + 배당 100원
    d_split, d_div = dates[240], dates[242]
    m = prices["ticker"] == t
    prices.loc[m & (prices["date"] < d_split), ["open", "close"]] /= 2
    prices.loc[m & (prices["date"] == d_split), "split"] = 2.0
    prices.loc[m & (prices["date"] == d_div), "dividend"] = 100.0
    run_days(comp, prices, dates, 240, 244)
    state, files = paper.load()
    tr = files["trades"]
    mine = tr[(tr["account"] == acc) & (tr["ticker"] == t)]
    assert (mine["side"] == "분할").any()
    assert not mine["reason"].astype(str).str.contains("손절").any()
    h = files["hold"][(files["hold"]["account"] == acc) & (files["hold"]["ticker"] == t)]
    if len(h):
        assert int(h["shares"].iloc[0]) == shares0 * 2
    div = mine[mine["side"] == "배당"]
    assert len(div) == 1
    assert div["amount"].iloc[0] == pytest.approx(shares0 * 2 * 100 * (1 - PAPER_DIV_TAX["KR"]), rel=1e-6)
    a = state["accounts"][acc]
    assert cash_from_trades(acc, a) == pytest.approx(a["cash"], abs=1.0)


def test_limit_up_open_blocks_buy(paper_dir):
    comp, prices, dates = make_market()
    run_days(comp, prices, dates, 230, 231)          # 개설 + 매수 주문
    _, files = paper.load()
    first = files["orders"][(files["orders"]["account"] == "KR-balanced") & (files["orders"]["side"] == "매수")]
    t = first.sort_values("rank")["ticker"].iloc[0]
    prev = prices[(prices["ticker"] == t) & (prices["date"] == dates[230])]["close"].iloc[0]
    prices.loc[(prices["ticker"] == t) & (prices["date"] == dates[231]), "open"] = prev * 1.30
    run_days(comp, prices, dates, 231, 232)
    _, files = paper.load()
    tr = files["trades"]
    assert not ((tr["account"] == "KR-balanced") & (tr["ticker"] == t) & (tr["side"] == "매수")).any()


def test_liquidity_cap_limits_position(paper_dir):
    comp, prices, dates = make_market()
    prices["volume"] = 300          # 거래가 아주 적은 시장
    run_days(comp, prices, dates, 230, 232)
    _, files = paper.load()
    buys = files["trades"][files["trades"]["side"] == "매수"]
    for _, b in buys.iterrows():
        adv = (prices[(prices["ticker"] == b["ticker"]) & (prices["date"] < b["date"])]
               .assign(v=lambda x: x["close"] * x["volume"])["v"].tail(20).mean())
        assert b["amount"] <= adv * 0.01 * 1.01


def test_v1_records_are_migrated(paper_dir):
    import json
    (paper_dir / "state.json").write_text(json.dumps({"KR": {
        "start_date": "2026-01-02", "start_cash": 50000000, "cash": 1000, "start_usdkrw": 1400, "currency": "KRW",
        "last_date": "2026-01-02", "last_rebalance_week": "2026-W01", "bench_start": 2500}}))
    pd.DataFrame([{"account": "KR", "ticker": "000010", "name": "회사1", "group": "금융", "shares": 10,
                   "avg_cost": 5000, "buy_date": "2026-01-02"}]).to_csv(paper_dir / "holdings.csv", index=False)
    pd.DataFrame([{"date": "2026-01-02", "account": "KR", "side": "매수", "ticker": "000010", "name": "회사1",
                   "shares": 10, "price": 5000, "amount": 50000, "cost": 7.5, "reason": "추천"}]
                 ).to_csv(paper_dir / "trades.csv", index=False)
    pd.DataFrame([{"date": "2026-01-02", "account": "KR", "cash": 1000, "stock_value": 50000, "nav": 51000,
                   "return_pct": 0.0, "holdings": 1}]).to_csv(paper_dir / "history.csv", index=False)
    state, files = paper.load()
    assert state["version"] == 2 and "KR-balanced" in state["accounts"]
    assert (files["trades"]["account"] == "KR-balanced").all()
    assert (files["trades"]["rules"] == "v1").all()
    assert files["hold"]["last_price"].iloc[0] == 5000


def test_v3_switch_keeps_old_balanced_as_legacy(paper_dir):
    import json
    comp, prices, dates = make_market()
    run_days(comp, prices, dates, 230, 233)
    st = json.loads((paper_dir / "state.json").read_text())
    # 예전(v3 전환 전) 상태를 흉내: 표시를 지우고 균형형을 balanced로 되돌림
    st.pop("score_v3")
    st["accounts"].pop("KR-legacy")
    json.dump(st, open(paper_dir / "state.json", "w"))
    before = pd.read_csv(paper.TRADES)
    n_bal = int((before["account"] == "KR-balanced").sum())
    run_days(comp, prices, dates, 233, 235)
    state, files = paper.load()
    acc = state["accounts"]
    assert acc["KR-balanced"]["strategy"] == "legacy"
    assert "KR-balanced-v3" in acc and acc["KR-balanced-v3"]["strategy"] == "balanced"
    assert (files["trades"]["account"] == "KR-balanced").sum() >= n_bal
    assert sum(a["strategy"] == "legacy" for a in acc.values()) == 1

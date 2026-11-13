"""Yahoo 가격 오류 정리 검사."""
import numpy as np
import pandas as pd
import pytest

import analyze
import paper
from conftest import make_market, flat_bench


def test_unrecorded_reverse_split_is_aligned():
    idx = [f"d{i:02d}" for i in range(10)]
    c = pd.DataFrame({"A": [1.0, 1.02, 0.99, 1.01, 10.1, 10.3, 9.9, 10.0, 10.2, 10.1]}, index=idx)
    out = analyze.clean_close(c)
    r = out["A"].pct_change().abs()
    assert r.max() < 0.2                       # 10배 점프가 사라짐
    assert out["A"].iloc[-1] == pytest.approx(10.1)


def test_one_day_bad_tick_is_removed():
    idx = [f"d{i:02d}" for i in range(8)]
    c = pd.DataFrame({"A": [100, 101, 99, 6250000, 100, 102, 101, 100.0]}, index=idx)
    out = analyze.clean_close(c)
    assert np.isnan(out["A"].iloc[3])
    assert out["A"].iloc[:3].tolist() == [100, 101, 99]


def test_recorded_split_day_is_left_alone():
    idx = [f"d{i:02d}" for i in range(6)]
    c = pd.DataFrame({"A": [10.0, 10.1, 1.0, 1.01, 1.02, 1.0]}, index=idx)   # Yahoo가 이미 보정 안 한 경우라도
    sp = pd.DataFrame({"A": [0, 0, 10.0, 0, 0, 0]}, index=idx)
    out = analyze.clean_close(c, sp)
    assert out["A"].iloc[0] == 10.0


def test_paper_treats_unrecorded_reverse_split_as_share_change(paper_dir):
    comp, prices, dates = make_market()
    from test_paper import run_days, cash_from_trades
    run_days(comp, prices, dates, 230, 236)
    state, files = paper.load()
    acc = "KR-balanced"
    h = files["hold"][files["hold"]["account"] == acc].iloc[0]
    t, sh, val0 = h["ticker"], int(h["shares"]), float(h["shares"] * h["last_price"])
    m = (prices["ticker"] == t) & (prices["date"] >= dates[236])
    prices.loc[m, ["open", "close"]] *= 10                    # 1:10 병합이 기록 없이 반영됨
    run_days(comp, prices, dates, 236, 237)
    state, files = paper.load()
    h2 = files["hold"][(files["hold"]["account"] == acc) & (files["hold"]["ticker"] == t)]
    assert len(h2) == 1
    assert int(h2["shares"].iloc[0]) == round(sh / 10)
    assert float(h2["shares"].iloc[0] * h2["last_price"].iloc[0]) == pytest.approx(val0, rel=0.25)
    a = state["accounts"][acc]
    assert cash_from_trades(acc, a) == pytest.approx(a["cash"], abs=1.0)


def test_fractional_split_on_integer_volume():
    """정수 거래량에 1.5:1 같은 분할 비율을 곱해도 멈추지 않아야 함 (2026-11-10~13 자동 분석 실패 원인)."""
    import yahoo_prices as yp
    old = pd.DataFrame({"date": ["2026-11-06", "2026-11-07"], "ticker": "A", "open": [150, 151], "close": [150, 152],
                        "volume": pd.Series([1001, 2003], dtype="int64"), "dividend": 0.0, "split": 0.0})
    new = pd.DataFrame({"date": ["2026-11-10"], "ticker": "A", "open": [101.0], "close": [101.5], "volume": [3000],
                        "dividend": 0.0, "split": [1.5]})
    out = yp._apply_new_splits(old, new)
    assert out["close"].tolist() == pytest.approx([100.0, 152 / 1.5])
    assert out["volume"].tolist() == pytest.approx([1501.5, 3004.5])

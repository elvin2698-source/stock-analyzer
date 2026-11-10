"""점수 검증·백테스트 검사."""
import numpy as np
import pandas as pd
import pytest

import analyze
import validate
from conftest import make_market


def test_backtest_tech_score_matches_daily_analysis():
    comp, prices, dates = make_market(n=80, days=260)
    res = analyze.analyze(prices, comp, "KR")
    adj = analyze.adjusted_close(prices)
    vol = analyze._wide(prices, "volume").reindex_like(adj)
    d = adj.index[-1]
    f = pd.DataFrame({"price": adj.loc[d], "ma20": adj.rolling(20).mean().loc[d], "ma60": adj.rolling(60).mean().loc[d],
                      "ma200": adj.rolling(200, min_periods=200).mean().loc[d], "ret_3m": (adj / adj.shift(63) - 1).loc[d],
                      "rsi": analyze.rsi_frame(adj).loc[d], "vol_ratio": (vol.rolling(5).mean() / vol.rolling(60).mean()).loc[d],
                      "ret_1w": (adj / adj.shift(5) - 1).loc[d] * 100})
    bt_score = sum(analyze.tech_components(f)).round(1)
    daily = res.set_index("ticker")["tech_score"]
    assert (bt_score.reindex(daily.index) - daily).abs().max() < 0.11


def test_validation_detects_predictive_score(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "SIG_DIR", tmp_path / "signals")
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2026-01-01", periods=120).date.astype(str)
    tick = [f"T{i}" for i in range(200)]
    skill = rng.normal(0, 1, 200)                       # 이 값이 높은 종목이 실제로 더 오름
    r = 0.002 * skill[None, :] + rng.normal(0, 0.01, (120, 200))
    adj = pd.DataFrame(np.exp(np.cumsum(r, 0)) * 100, index=dates, columns=tick)
    for d in dates[:50:5]:
        (tmp_path / "signals" / "KR").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": tick, "score": skill * 10 + 50, "tech_score": skill * 5 + 25,
                      "value_score": rng.normal(25, 5, 200), "grade": np.where(skill > 0.8, "추천", "중립")}
                     ).to_csv(tmp_path / "signals" / "KR" / f"{d}.csv.gz", index=False, compression="gzip")
    out = validate.validate("KR", adj)
    h = out["horizons"]["20"]
    assert h["ic_mean"] > 0.3
    assert h["quintiles"][4] > h["quintiles"][0]
    assert h["grades"]["추천"] > h["grades"]["중립"]
    assert abs(h["ic_value"]) < 0.15


def test_backtest_runs_and_returns_curves():
    comp, prices, dates = make_market(n=120, days=300)
    res = analyze.analyze(prices, comp, "KR")
    out = validate.backtest("KR", prices, res)
    assert out and len(out["dates"]) == len(out["strategy"]) == len(out["ew"])
    assert out["stats"]["strategy"]["mdd"] <= 0

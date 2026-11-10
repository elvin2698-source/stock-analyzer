"""점수 검증·백테스트 검사."""
import numpy as np
import pandas as pd
import pytest

import analyze
import validate
from conftest import make_market


def test_backtest_scores_match_daily_analysis():
    """백테스트가 오늘 분석과 똑같은 기술점수 계산(v3·v2 모두)을 쓰는지."""
    comp, prices, dates = make_market(n=80, days=300)
    res = analyze.analyze(prices, comp, "KR")
    F = validate._features(prices, res)
    d = F["adj"].index[-1]
    cols = F["adj"].columns
    daily = res.set_index("ticker")
    v3 = validate._score_at(F, d, cols, "v3").round(1)
    v2 = validate._score_at(F, d, cols, "v2").round(1)
    assert (v3.reindex(daily.index) - daily["tech_score"]).abs().max() < 0.11
    assert (v2.reindex(daily.index) - daily["tech_score_v2"]).abs().max() < 0.11


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


def test_compare_scores_returns_both_versions():
    comp, prices, dates = make_market(n=120, days=330)
    res = analyze.analyze(prices, comp, "KR")
    out = validate.compare_scores("KR", prices, res)
    for v in ("v2", "v3"):
        x = out["versions"][v]
        assert len(x["curve"]) == len(out["dates"]) == len(out["ew_curve"])
        assert x["mdd"] <= 0 and len(x["quintiles"]) == 5
    assert set(out["ic_better"]) == {"A", "B"}

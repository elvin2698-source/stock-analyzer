"""과거 재무(시점별 EPS·BPS)와 종합점수 백테스트 검사."""
import numpy as np
import pandas as pd
import pytest

import analyze
import fundamentals
import validate
from conftest import make_market


def rec(t, period, kind, eps, bps, market="KR"):
    from datetime import date
    return fundamentals._rows(t, date.fromisoformat(period), kind, eps, bps, market)


def test_no_future_information():
    f = pd.DataFrame([rec("A", "2025-12-31", "A", 1000, 10000), rec("A", "2024-12-31", "A", 500, 9000)])
    # 2025년 연간 실적은 결산 90일 뒤(2026-03-31)부터 사용
    assert fundamentals.at(f, "2026-03-30").loc["A", "eps"] == 500
    assert fundamentals.at(f, "2026-03-31").loc["A", "eps"] == 1000
    assert fundamentals.at(f, "2025-01-01").empty


def test_ttm_from_four_quarters_beats_annual():
    rows = [rec("A", "2025-12-31", "A", 400, 5000)]
    for p, e in (("2025-12-31", 100), ("2026-03-31", 110), ("2026-06-30", 120), ("2026-09-30", 130)):
        rows.append(rec("A", p, "Q", e, 6000))
    f = pd.DataFrame(rows)
    x = fundamentals.at(f, "2026-11-20")              # 3분기 실적(9/30)은 45일 뒤 11/14부터
    assert x.loc["A", "eps"] == 460
    assert x.loc["A", "bps"] == 6000
    y = fundamentals.at(f, "2026-11-10")              # 아직 3분기 미발표 → 분기 3개뿐 → 연간 사용
    assert y.loc["A", "eps"] == 400


def test_naver_parsing(monkeypatch):
    payload = {"financeInfo": {"trTitleList": [{"key": "202412", "isConsensus": "N"}, {"key": "202612", "isConsensus": "Y"}],
                               "rowList": [{"title": "EPS", "columns": {"202412": {"value": "4,950"}, "202612": {"value": "9,999"}}},
                                           {"title": "BPS", "columns": {"202412": {"value": "57,981"}, "202612": {"value": "1"}}}]}}

    class R:
        def json(self):
            return payload
    monkeypatch.setattr(fundamentals.requests, "get", lambda *a, **k: R())
    rows = fundamentals.fetch_kr_one("005930")
    assert {(r["kind"], r["period"], r["eps"]) for r in rows} == {("A", "2024-12-31", 4950.0), ("Q", "2024-12-31", 4950.0)}
    assert all(r["available"] > r["period"] for r in rows)


def test_compare_scores_includes_value_and_total():
    comp, prices, dates = make_market(n=120, days=330)
    res = analyze.analyze(prices, comp, "KR")
    rng = np.random.default_rng(0)
    fund = pd.DataFrame([rec(t, "2025-06-30", "A", float(rng.uniform(-500, 3000)), float(rng.uniform(5000, 40000)))
                         for t in comp["ticker"]])
    out = validate.compare_scores("KR", prices, res, fund=fund)
    assert {"total", "total_v2", "value", "v3", "v2"} <= set(out["versions"])
    assert out["value_coverage"] > 50
    t = out["versions"]["total"]
    assert len(t["curve"]) == len(out["dates"])


def test_naver_without_finance_info_is_skipped(monkeypatch):
    class R:
        def json(self):
            return {"financeInfo": None}
    monkeypatch.setattr(fundamentals.requests, "get", lambda *a, **k: R())
    assert fundamentals.fetch_kr_one("123450") == []

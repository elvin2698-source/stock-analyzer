import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def paper_dir(tmp_path, monkeypatch):
    """모의투자 파일을 임시 폴더에 쓰도록 바꿈."""
    import paper
    d = tmp_path / "paper"
    d.mkdir()
    monkeypatch.setattr(paper, "PAPER_DIR", d)
    for name, fn in (("STATE", "state.json"), ("HOLD", "holdings.csv"), ("ORDERS", "orders.csv"),
                     ("TRADES", "trades.csv"), ("HIST", "history.csv"), ("BENCH", "bench.csv"), ("EW", "ew.csv")):
        monkeypatch.setattr(paper, name, d / fn)
    return d


def make_market(n=120, days=280, seed=0, start="2025-09-01"):
    """가짜 시장: 회사 목록 + 주가(시가·종가·거래량·배당·분할)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days).date.astype(str)
    tick = [f"{i:05d}0" for i in range(n)]
    r = rng.normal(0.0006, 0.02, (days, n))
    close = np.exp(np.cumsum(r, 0)) * rng.uniform(2000, 60000, n)
    opn = close * np.exp(rng.normal(0, 0.005, (days, n)))
    vol = rng.integers(20_000, 2_000_000, (days, n))
    prices = pd.DataFrame({"date": np.repeat(dates, n), "ticker": np.tile(tick, days),
                           "open": opn.ravel(), "close": close.ravel(), "volume": vol.ravel(),
                           "dividend": 0.0, "split": 0.0})
    groups = ["반도체·전자부품", "바이오·제약", "금융", "화학·소재", "자동차·부품", "소프트웨어·인터넷"]
    comp = pd.DataFrame({"ticker": tick, "name": [f"회사{i}" for i in range(n)], "market": "KOSPI",
                         "sector": [groups[i % 6] for i in range(n)], "per": rng.uniform(3, 30, n),
                         "pbr": rng.uniform(0.3, 3, n), "roe": rng.uniform(-5, 25, n),
                         "div_yield": rng.uniform(0, 5, n), "market_cap": rng.uniform(5e10, 5e12, n)})
    return comp, prices, list(dates)


def flat_bench(market):
    return pd.DataFrame(columns=["date", "symbol", "close"])

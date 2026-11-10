"""점검용: 백테스트가 왜 나쁜지, 하락장 판단이 맞는지, 이상한 가격이 있는지 확인."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
import analyze, categorize
from config import read_csv, MIN_TRADING_VALUE
pd.set_option("display.width", 250)
b = read_csv("paper/bench.csv")
for sym in ("^KS11", "^GSPC"):
    s = b[b.symbol == sym].set_index("date")["close"].astype(float).sort_index()
    print(sym, "last", s.index[-1], round(s.iloc[-1], 1), "ma200", round(s.tail(200).mean(), 1), "n", len(s), "max", round(s.max(), 1), s.idxmax())
for m in ("kr", "us"):
    p = read_csv(f"data/prices_{m}.v2.csv.gz")
    comp = read_csv(f"data/companies_{m}.csv")
    adj = analyze.adjusted_close(p)
    raw = analyze._wide(p, "close")
    r = adj.pct_change()
    print(f"\n===== {m} prices", p.shape, "dates", adj.index[0], adj.index[-1], "tickers", adj.shape[1])
    print("daily |ret|>40%:", int((r.abs() > 0.4).sum().sum()), " >100%:", int((r > 1).sum().sum()))
    big = r.stack()
    big = big[big.abs() > 0.6].sort_values()
    print("extreme moves sample:\n", big.head(8).to_string(), "\n", big.tail(8).to_string())
    # 되돌림 스파이크: 하루 +50% 후 다음날 -33% 이상
    spike = ((r > 0.5) & (r.shift(-1) < -0.33)).sum().sum()
    print("spike-and-revert count:", int(spike))
    ev = p[(p.dividend > 0) | (p.split > 0)]
    print("events: dividends", int((p.dividend > 0).sum()), "splits", int((p.split > 0).sum()))
    divratio = (ev.dividend / ev.close).describe()
    print("dividend/close:", divratio.round(4).to_dict())
    f = adj.iloc[0] / raw.iloc[0]
    print("adj factor first-day quantiles:", f.quantile([0.01, 0.1, 0.5, 0.9]).round(3).to_dict())
    # 백테스트 분해: 비용 없이 / 상위 20 수익 기여
    vol = analyze._wide(p, "volume").reindex_like(adj)
    ma20, ma60, ma200 = adj.rolling(20).mean(), adj.rolling(60).mean(), adj.rolling(200, min_periods=200).mean()
    rsi = analyze.rsi_frame(adj); r1, r3 = adj / adj.shift(5) - 1, adj / adj.shift(63) - 1
    vr = vol.rolling(5).mean() / vol.rolling(60).mean(); tv = (adj * vol).rolling(20).mean(); days = adj.notna().cumsum()
    dates = list(adj.index); eq, held, contrib = 1.0, [], {}
    worst = []
    for i in range(70, len(dates) - 1):
        d = dates[i]
        if (i - 70) % 5 == 0:
            el = (tv.loc[d] >= MIN_TRADING_VALUE[m.upper()]) & (days.loc[d] >= 60) & adj.loc[d].notna()
            c = el[el].index
            fr = pd.DataFrame({"price": adj.loc[d, c], "ma20": ma20.loc[d, c], "ma60": ma60.loc[d, c], "ma200": ma200.loc[d, c],
                               "ret_3m": r3.loc[d, c], "rsi": rsi.loc[d, c], "vol_ratio": vr.loc[d, c], "ret_1w": r1.loc[d, c] * 100})
            sc = sum(analyze.tech_components(fr))
            held = list(sc.sort_values(ascending=False).head(20).index)
            if i == 70 or i % 100 == 0:
                print(d, "top score", round(sc.max(), 1), "n with max", int((sc >= sc.max() - 1e-9).sum()), "elig", len(c))
        nd = dates[i + 1]
        rr = r.loc[nd, held].fillna(0).clip(-0.5, 0.5)
        eq *= 1 + rr.mean()
        for t, v in rr.items():
            contrib[t] = contrib.get(t, 0) + v / 20
        worst.append((nd, rr.mean()))
    print("no-cost equity", round(eq, 3))
    cs = pd.Series(contrib).sort_values()
    print("worst contributors:\n", cs.head(10).round(3).to_string())
    print("best contributors:\n", cs.tail(5).round(3).to_string())
    w = sorted(worst, key=lambda x: x[1])[:6]
    print("worst days:", [(d, round(v * 100, 1)) for d, v in w])
    t0 = cs.index[0]
    s = raw[t0].dropna()
    print("worst ticker", t0, "raw close sample:\n", s.iloc[::max(1, len(s) // 25)].round(2).to_string())

"""점수 검증과 백테스트.

1) 점수 검증 (앞으로 쌓이는 실제 기록)
   매일 분석이 끝나면 전 종목의 점수를 paper/signals/<시장>/<날짜>.csv.gz 로 남깁니다.
   며칠·몇 주가 지나 실제 수익률이 나오면, 점수가 높은 종목일수록 실제로 더 올랐는지 확인합니다.
     - IC(정보계수): 그날 점수 순위와 이후 수익률 순위의 상관(-1~1). 0.03~0.05만 꾸준해도 쓸모 있는 수준
     - 5분위: 점수로 종목을 5묶음으로 나눠 이후 평균 수익률 비교 (5분위가 1분위보다 높아야 함)
     - 등급별: 추천·중립·비추천의 이후 평균 수익률
2) 백테스트 (과거 2년 주가로 바로 확인)
   과거 PER·PBR 기록은 무료로 구할 수 없어 '기술점수'만 검증합니다.
   매주 기술점수 상위 20종목을 같은 금액으로 사서 1주 보유하고, 비용을 뺀 성과를 전 종목 동일가중과 비교합니다.
   주의: 지금 상장된 종목만으로 계산해 상장폐지 종목이 빠져 있어(생존자 편향) 실제보다 좋게 나올 수 있습니다.
"""
import json

import numpy as np
import pandas as pd

import analyze
from config import PAPER_DIR, PAPER_COSTS, PAPER_SLIPPAGE, MIN_TRADING_VALUE, read_csv

SIG_DIR = PAPER_DIR / "signals"
VALID = PAPER_DIR / "validation.json"
BACKTEST = PAPER_DIR / "backtest.json"
HORIZONS = {5: "1주", 20: "1개월", 60: "3개월"}


def save_signals(market: str, res: pd.DataFrame):
    d = str(res["as_of"].iloc[0])
    out = SIG_DIR / market / f"{d}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ticker", "score", "tech_score", "value_score", "grade"]
    res.loc[res["grade"] != "제외", cols].to_csv(out, index=False, compression="gzip")


def _spearman(a: pd.Series, b: pd.Series):
    m = a.notna() & b.notna()
    if m.sum() < 30:
        return None
    return float(a[m].rank().corr(b[m].rank()))


def _summary(ics, quint_rows, grade_rows, n_obs):
    ics = [x for x in ics if x is not None]
    if not ics:
        return None
    ics = np.array(ics)
    q = pd.DataFrame(quint_rows).mean() if quint_rows else pd.Series(dtype=float)
    g = pd.DataFrame(grade_rows).mean() if grade_rows else pd.Series(dtype=float)
    return {"n_dates": int(len(ics)), "n_obs": int(n_obs), "ic_mean": round(float(ics.mean()), 4),
            "ic_t": round(float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))), 2) if len(ics) > 2 and ics.std(ddof=1) > 0 else None,
            "ic_hit": round(float((ics > 0).mean()) * 100, 1),
            "quintiles": [None if pd.isna(q.get(i)) else round(float(q[i]) * 100, 3) for i in range(1, 6)],
            "grades": {k: round(float(v) * 100, 3) for k, v in g.items() if pd.notna(v)}}


def validate(market: str, adj: pd.DataFrame) -> dict:
    files = sorted((SIG_DIR / market).glob("*.csv.gz")) if (SIG_DIR / market).exists() else []
    idx = {d: i for i, d in enumerate(adj.index)}
    out = {"snapshots": len(files), "first": files[0].name[:10] if files else None, "horizons": {}}
    for h, label in HORIZONS.items():
        fwd = adj.shift(-h) / adj - 1
        ics, ics_t, ics_v, qrows, grows, n_obs = [], [], [], [], [], 0
        for f in files:
            d = f.name[:10]
            if d not in idx or idx[d] + h >= len(adj.index):
                continue                                   # 아직 h일이 지나지 않음
            s = read_csv(f).set_index("ticker")
            r = fwd.loc[d].reindex(s.index)
            ok = r.notna()
            if ok.sum() < 30:
                continue
            s, r = s[ok], r[ok].clip(-0.9, 3)
            n_obs += len(s)
            ics.append(_spearman(s["score"], r))
            ics_t.append(_spearman(s["tech_score"], r))
            ics_v.append(_spearman(s["value_score"], r))
            q = pd.qcut(s["score"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
            qrows.append(r.groupby(q, observed=True).mean().to_dict())
            grows.append(r.groupby(s["grade"]).mean().to_dict())
        summ = _summary(ics, qrows, grows, n_obs)
        if summ:
            t, v = [x for x in ics_t if x is not None], [x for x in ics_v if x is not None]
            summ["ic_tech"] = round(float(np.mean(t)), 4) if t else None
            summ["ic_value"] = round(float(np.mean(v)), 4) if v else None
            summ["label"] = label
            out["horizons"][str(h)] = summ
    return out


def backtest(market: str, prices: pd.DataFrame, res: pd.DataFrame, n=20, every=5) -> dict:
    adj = analyze.adjusted_close(prices)
    vol = analyze._wide(prices, "volume").reindex_like(adj)
    ok_kind = set(res.loc[res.get("instrument", pd.Series("", index=res.index)).fillna("") == "", "ticker"])
    adj = adj[[c for c in adj.columns if c in ok_kind]]
    vol = vol[adj.columns]
    if len(adj) < 130:
        return {}
    ma20, ma60 = adj.rolling(20).mean(), adj.rolling(60).mean()
    ma200 = adj.rolling(200, min_periods=200).mean()
    rsi = analyze.rsi_frame(adj)
    ret1w, ret3m = adj / adj.shift(5) - 1, adj / adj.shift(63) - 1
    vratio = vol.rolling(5).mean() / vol.rolling(60).mean()
    tv = (adj * vol).rolling(20).mean()
    days = adj.notna().cumsum()
    daily = adj.pct_change().clip(-0.5, 0.5)
    cost = PAPER_COSTS[market]
    fee_in, fee_out = cost["buy"] + PAPER_SLIPPAGE, cost["sell"] + PAPER_SLIPPAGE

    dates = list(adj.index)
    start = 70
    held, prev = [], set()
    strat, ew, curve_d = [1.0], [1.0], [dates[start]]
    quint, ics = [], []
    for i in range(start, len(dates) - 1):
        d = dates[i]
        if (i - start) % every == 0:
            elig = (tv.loc[d] >= MIN_TRADING_VALUE[market]) & (days.loc[d] >= 60) & adj.loc[d].notna()
            cols = elig[elig].index
            if len(cols) < n * 3:
                continue
            f = pd.DataFrame({"price": adj.loc[d, cols], "ma20": ma20.loc[d, cols], "ma60": ma60.loc[d, cols],
                              "ma200": ma200.loc[d, cols], "ret_3m": ret3m.loc[d, cols], "rsi": rsi.loc[d, cols],
                              "vol_ratio": vratio.loc[d, cols], "ret_1w": ret1w.loc[d, cols] * 100})
            score = sum(analyze.tech_components(f))
            new = list(score.sort_values(ascending=False).head(n).index)
            turn_out = len(prev - set(new)) / n if prev else 0
            turn_in = len(set(new) - prev) / n
            strat[-1] *= 1 - (turn_out * fee_out + turn_in * fee_in)
            held, prev = new, set(new)
            if i + 20 < len(dates):
                fwd = (adj.iloc[i + 20][cols] / adj.loc[d, cols] - 1).clip(-0.9, 3)
                ics.append(_spearman(score, fwd))
                q = pd.qcut(score.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
                quint.append(fwd.groupby(q, observed=True).mean().to_dict())
            universe = cols
        if not held:
            continue
        nd = dates[i + 1]
        strat.append(strat[-1] * (1 + float(daily.loc[nd, held].fillna(0).mean())))
        ew.append(ew[-1] * (1 + float(daily.loc[nd, universe].dropna().mean())))
        curve_d.append(nd)

    def stats(c):
        c = pd.Series(c, index=curve_d[:len(c)])
        r = c.pct_change().dropna()
        yrs = max(len(r) / 252, 1e-9)
        vol_ = float(r.std() * np.sqrt(252)) if len(r) > 1 else None
        return {"total": round((c.iloc[-1] - 1) * 100, 2), "cagr": round((c.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
                "mdd": round(float((c / c.cummax() - 1).min()) * 100, 2),
                "vol": round(vol_ * 100, 2) if vol_ else None,
                "sharpe": round((float(r.mean()) * 252 - 0.03) / vol_, 2) if vol_ else None}

    if len(strat) < 30:
        return {}
    step = max(1, len(curve_d) // 150)                     # 그래프용으로 150점 정도만
    pick = list(range(0, len(curve_d), step)) + [len(curve_d) - 1]
    qm = pd.DataFrame(quint).mean() if quint else pd.Series(dtype=float)
    icv = [x for x in ics if x is not None]
    return {"period": [curve_d[0], curve_d[-1]], "n": n, "every": every,
            "dates": [curve_d[j] for j in pick], "strategy": [round((strat[j] - 1) * 100, 2) for j in pick],
            "ew": [round((ew[j] - 1) * 100, 2) for j in pick],
            "stats": {"strategy": stats(strat), "ew": stats(ew)},
            "quintiles_20d": [None if pd.isna(qm.get(k)) else round(float(qm[k]) * 100, 3) for k in range(1, 6)],
            "ic_20d": round(float(np.mean(icv)), 4) if icv else None, "n_rebalances": len(quint)}


def run_all(prices_by_market: dict, results: dict):
    """각 시장의 검증·백테스트를 계산해 paper/에 저장하고 대시보드용으로 돌려줍니다."""
    PAPER_DIR.mkdir(exist_ok=True)
    val, bt = {}, {}
    for m, prices in prices_by_market.items():
        try:
            adj = analyze.adjusted_close(prices)
            val[m] = validate(m, adj)
            bt[m] = backtest(m, prices, results[m])
            s = bt[m].get("stats", {})
            if s:
                print(f"  [검증] {m} 백테스트(기술점수 상위 20): {s['strategy']['total']:+.1f}% vs 동일가중 {s['ew']['total']:+.1f}%")
        except Exception as e:
            print(f"  [검증] {m} 실패: {type(e).__name__}: {e}")
    VALID.write_text(json.dumps(val, ensure_ascii=False), encoding="utf-8")
    BACKTEST.write_text(json.dumps(bt, ensure_ascii=False), encoding="utf-8")
    return {"validation": val, "backtest": bt}

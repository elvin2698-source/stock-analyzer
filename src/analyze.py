"""기술적 분석 + 가치 분석으로 종목별 점수(0~100)와 등급을 계산합니다.

[기술적 분석 50점]
  추세   20점: 현재가 > 200일선(8), 20일선 > 60일선(7), 현재가 > 20일선(5)
  모멘텀 15점: 최근 3개월 수익률이 같은 시장 안에서 상위일수록 높음
  RSI    10점: 40~60 안정(10), 30~40·60~70(7), 30 미만 과매도(5), 70 초과 과열(2)
  거래량  5점: 거래량 급증 + 상승(5), 보통(2.5), 거래량 급증 + 하락(0)
[가치 분석 50점]  같은 업종 안에서 비교 (업종 종목이 적으면 시장 전체와 비교)
  PER 15점: 낮을수록 높음 (적자면 0점)
  PBR 10점: 낮을수록 높음
  ROE 15점: 높을수록 높음 (0 이하면 0점)
  배당  5점: 배당수익률이 높을수록 높음
  재무 데이터가 없으면 해당 항목은 절반 점수
"""
import numpy as np
import pandas as pd

from config import BUY_SCORE, SELL_SCORE, MIN_TRADING_VALUE
import categorize


def _wide(prices: pd.DataFrame, col: str) -> pd.DataFrame:
    prices = prices.drop_duplicates(["date", "ticker"], keep="last")   # 혹시 겹친 줄이 있어도 멈추지 않게
    return prices.pivot(index="date", columns="ticker", values=col).sort_index()


JUMP = 3.0          # 하루에 3배 이상 오르거나 1/3 이하로 떨어지면 데이터 이상으로 의심


def clean_close(close: pd.DataFrame, splits: pd.DataFrame = None) -> pd.DataFrame:
    """Yahoo 데이터에 섞인 두 가지 오류를 고칩니다.
    - 기록되지 않은 액면병합·분할: 하루에 3배↑ 또는 1/3↓로 바뀐 뒤 그 수준이 유지되면, 그 이전 가격을 같은 비율로 맞춤
    - 하루짜리 잘못된 가격: 튀었다가 바로 원래 수준으로 돌아오면 그날 가격을 지움
    기록된 액면분할이 있는 날은 건드리지 않습니다."""
    c = close.copy()
    prev = c.ffill().shift(1)
    ratio = c / prev
    sus = (ratio >= JUMP) | (ratio <= 1 / JUMP)
    if splits is not None:
        sus &= ~(splits.reindex_like(c).fillna(0) > 0)
    for t in sus.columns[sus.any()]:
        col = c[t]
        for d in sus.index[sus[t]]:
            i = c.index.get_loc(d)
            before = col.iloc[:i].dropna()
            after = col.iloc[i + 1:i + 4].dropna()
            if before.empty or pd.isna(col.iloc[i]):
                continue
            level = col.iloc[i]
            if len(after) and (after / before.iloc[-1]).between(1 / 1.6, 1.6).all():
                col.iloc[i] = np.nan                                   # 하루만 튄 값 → 지움
            else:
                col.iloc[:i] = col.iloc[:i] * (level / before.iloc[-1])  # 기록 안 된 분할·병합 → 과거 가격 보정
        c[t] = col
    return c


def adjusted_close(prices: pd.DataFrame) -> pd.DataFrame:
    """배당까지 반영한 수정주가 (날짜 × 종목). 저장된 가격은 액면분할만 반영돼 있으므로
    배당락일마다 (1 - 배당금/전날 종가)를 그 이전 가격 전체에 곱해 줍니다.
    먼저 clean_close()로 기록되지 않은 분할·병합과 하루짜리 오류 가격을 정리합니다."""
    close = _wide(prices, "close")
    if "split" in prices:
        close = clean_close(close, _wide(prices, "split").reindex_like(close))
    if "dividend" not in prices:
        return close
    div = _wide(prices, "dividend").reindex_like(close).fillna(0)
    prev = close.ffill().shift(1)
    ratio = (1 - div / prev).where(div > 0, 1.0).clip(lower=0.5).fillna(1.0)   # 비정상 값 방어
    factor = ratio.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return close * factor


def rsi_frame(close: pd.DataFrame, n=14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def rsi(close: pd.DataFrame, n=14) -> pd.Series:
    return rsi_frame(close, n).iloc[-1]


def tech_components(df: pd.DataFrame):
    """기술점수 4요소. 오늘 분석과 백테스트가 같은 계산을 쓰도록 한 곳에 둡니다.
    df 열: price, ma20, ma60, ma200, ret_3m, rsi, vol_ratio, ret_1w"""
    above200 = df["price"] > df["ma200"]
    golden = df["ma20"] > df["ma60"]
    above20 = df["price"] > df["ma20"]
    trend = above200 * 8 + golden * 7 + above20 * 5
    trend = trend.where(df["ma200"].notna(), golden * 10 + above20 * 5)   # 상장 200일 미만
    mom = df["ret_3m"].rank(pct=True).fillna(0.5) * 15
    r = df["rsi"]
    rsi_pts = pd.Series(np.select([r.between(40, 60), r.between(30, 70), r < 30, r > 70], [10, 7, 5, 2], 5), index=df.index)
    surge = df["vol_ratio"] > 1.5
    vol_pts = pd.Series(np.select([surge & (df["ret_1w"] > 0), surge & (df["ret_1w"] <= 0)], [5, 0], 2.5), index=df.index)
    return trend, mom, rsi_pts, vol_pts


def technicals(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    last = close.ffill().iloc[-1]
    t = pd.DataFrame({
        "price": last,
        "ma20": close.rolling(20).mean().iloc[-1],
        "ma60": close.rolling(60).mean().iloc[-1],
        "ma200": close.rolling(200, min_periods=200).mean().iloc[-1],
        "rsi": rsi(close),
        "ret_1w": last / close.shift(5).iloc[-1] - 1,
        "ret_3m": last / close.shift(63).iloc[-1] - 1,
        "vol_ratio": volume.iloc[-5:].mean() / volume.iloc[-60:].mean(),
        "trading_value": (close * volume).iloc[-20:].mean(),
        "days": close.notna().sum(),
    })
    t["ret_1w"] *= 100
    t["ret_3m"] *= 100
    return t


def _rank(s: pd.Series, groups: pd.Series, ascending=True) -> pd.Series:
    """업종 내 백분위 (0~1). 업종 종목이 5개 미만이면 시장 전체 기준."""
    overall = s.rank(pct=True, ascending=ascending)
    size = s.groupby(groups).transform("count")
    by_group = s.groupby(groups).rank(pct=True, ascending=ascending)
    return by_group.where(size >= 5, overall)


def score(df: pd.DataFrame, market: str) -> pd.DataFrame:
    df = df.copy()
    sector = df["sector"].fillna("기타")
    reasons = [[] for _ in range(len(df))]

    def note(mask, text):
        for i in np.flatnonzero(mask.fillna(False).to_numpy()):
            reasons[i].append(text)

    # ── 기술적 분석 ─────────────────────────────
    trend, mom, rsi_pts, vol_pts = tech_components(df)
    above200 = df["price"] > df["ma200"]
    golden = df["ma20"] > df["ma60"]
    note(above200, "200일선 위(장기 상승추세)")
    note(df["ma200"].notna() & ~above200, "200일선 아래(장기 하락추세)")
    note(golden, "20일선>60일선(단기 상승)")
    note(~golden & df["ma60"].notna(), "20일선<60일선(단기 하락)")

    note(mom >= 12, "3개월 수익률 상위 20%")
    note(mom <= 3, "3개월 수익률 하위 20%")

    r = df["rsi"]
    note(r < 30, "RSI 과매도(반등 가능)")
    note(r > 70, "RSI 과열(단기 조정 주의)")

    surge = df["vol_ratio"] > 1.5
    note(surge & (df["ret_1w"] > 0), "거래량 급증 + 상승")
    note(surge & (df["ret_1w"] <= 0), "거래량 급증 + 하락")

    df["tech_score"] = trend + mom + rsi_pts + vol_pts

    # ── 가치 분석 ───────────────────────────────
    per_pts = (_rank(df["per"], sector, ascending=False) * 15)
    per_pts = per_pts.where(df["per"].notna(), np.where(df["roe"] <= 0, 0, 7.5))   # PER 없음: 적자면 0, 모르면 중립
    note(per_pts >= 12, "업종 대비 PER 낮음(저평가)")
    note(df["per"].notna() & (per_pts <= 3), "업종 대비 PER 높음(고평가)")

    pbr_pts = (_rank(df["pbr"], sector, ascending=False) * 10).fillna(5)
    note(df["pbr"].notna() & (pbr_pts >= 8), "업종 대비 PBR 낮음")

    roe_pts = (_rank(df["roe"], sector) * 15).where(df["roe"] > 0, 0)
    roe_pts = roe_pts.where(df["roe"].notna(), 7.5)
    note(df["roe"] >= 15, "ROE 15% 이상(수익성 우수)")
    note(df["roe"] <= 0, "ROE 0 이하(적자)")

    div_pts = (_rank(df["div_yield"].where(df["div_yield"] > 0), sector) * 5).fillna(0)
    div_pts = div_pts.where(df["div_yield"].notna(), 2.5)
    note(df["div_yield"] >= 4, "배당수익률 4% 이상")

    note(df["per"].isna() & df["pbr"].isna() & df["roe"].isna(), "재무 데이터 없음(가치 점수 중립 처리)")
    df["value_score"] = per_pts + pbr_pts + roe_pts + div_pts
    df["styles"] = categorize.styles(df, per_pts)

    df["score"] = (df["tech_score"] + df["value_score"]).round(1)
    df["tech_score"] = df["tech_score"].round(1)
    df["value_score"] = df["value_score"].round(1)

    grade = np.select([df["score"] >= BUY_SCORE, df["score"] <= SELL_SCORE], ["추천", "비추천"], "중립")
    illiquid = df["trading_value"].fillna(0) < MIN_TRADING_VALUE[market]
    short = df["days"] < 60
    kind = df.get("instrument", pd.Series("", index=df.index)).fillna("")
    not_stock = kind != ""
    grade = np.where(not_stock | illiquid | short, "제외", grade)
    for k in sorted(set(kind) - {""}):
        note(kind == k, f"{k}(일반 기업 아님)으로 제외")
    note(illiquid & ~not_stock, "거래대금 부족으로 제외")
    note(short & ~illiquid & ~not_stock, "상장 60거래일 미만으로 제외")
    df["grade"] = grade
    df["reasons"] = [" · ".join(x) for x in reasons]
    return df


def analyze(prices: pd.DataFrame, companies: pd.DataFrame, market: str) -> pd.DataFrame:
    close = adjusted_close(prices)
    volume = _wide(prices, "volume").reindex_like(close)
    tech = technicals(close, volume)
    comp = companies.set_index("ticker")
    comp = comp.drop(columns=[c for c in tech.columns if c in comp.columns])
    df = comp.join(tech, how="inner")
    for c in ("per", "pbr", "roe", "div_yield"):
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["group"] = categorize.sector_group(df, market)
    df["size"] = categorize.size_class(df, market)
    df["instrument"] = categorize.instrument_kind(df, market)
    df = score(df, market)
    df["country"] = market
    df["as_of"] = close.index[-1]
    return df.reset_index().rename(columns={"index": "ticker"}).sort_values("score", ascending=False)

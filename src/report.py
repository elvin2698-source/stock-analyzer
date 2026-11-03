"""분석 결과를 엑셀(results/) + 웹 대시보드(docs/index.html) + CSV로 저장합니다."""
import json
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, DOCS_DIR, BUY_SCORE, SELL_SCORE
from categorize import GROUPS, STYLES

COLS = {"ticker": "종목코드", "name": "기업명", "country": "국가", "market": "시장",
        "group": "업종대분류", "sector": "세부업종", "size": "규모", "styles": "스타일",
        "grade": "등급", "score": "종합점수", "tech_score": "기술점수", "value_score": "가치점수",
        "price": "현재가", "ret_1w": "1주수익률(%)", "ret_3m": "3개월수익률(%)", "rsi": "RSI",
        "per": "PER", "pbr": "PBR", "roe": "ROE(%)", "div_yield": "배당수익률(%)",
        "market_cap": "시가총액", "reasons": "판단 근거"}
MARKET_KO = {"KOSPI": "코스피", "KOSDAQ": "코스닥", "NASDAQ": "나스닥", "NYSE": "뉴욕", "AMEX": "아멕스", "US": "미국"}


def _table(df):
    out = df[[c for c in COLS if c in df]].copy()
    out["country"] = out["country"].map({"KR": "한국", "US": "미국"})
    out["market"] = out["market"].map(MARKET_KO).fillna(out["market"])
    out = out.rename(columns=COLS)
    for c in ("1주수익률(%)", "3개월수익률(%)", "RSI", "PER", "PBR", "ROE(%)", "배당수익률(%)"):
        if c in out:
            out[c] = out[c].round(2)
    return out


def _group_summary(res):
    rated = res[res["grade"] != "제외"].reset_index(drop=True)
    g = rated.groupby(["country", "group"])
    out = pd.DataFrame({
        "종목수": g.size(),
        "추천": g["grade"].apply(lambda s: (s == "추천").sum()),
        "중립": g["grade"].apply(lambda s: (s == "중립").sum()),
        "비추천": g["grade"].apply(lambda s: (s == "비추천").sum()),
        "평균점수": g["score"].mean().round(1),
        "최고점 종목": rated.loc[g["score"].idxmax(), ["country", "group", "name"]].set_index(["country", "group"])["name"],
    }).reset_index()
    out["추천비율(%)"] = (out["추천"] / out["종목수"] * 100).round(1)
    out["country"] = out["country"].map({"KR": "한국", "US": "미국"})
    return out.rename(columns={"country": "국가", "group": "업종대분류"}).sort_values(["국가", "평균점수"], ascending=[True, False])


def write_excel(res: pd.DataFrame, as_of: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"추천결과_{as_of}.xlsx"
    rated = res[res["grade"] != "제외"]
    by_group = lambda d, asc: d.sort_values(["country", "group", "score"], ascending=[True, True, asc])
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary = res.groupby(["country", "grade"]).size().unstack(fill_value=0)
        summary.index = summary.index.map({"KR": "한국", "US": "미국"})
        summary.to_excel(xw, sheet_name="요약")
        _group_summary(res).to_excel(xw, sheet_name="업종별 요약", index=False)
        for name, part, asc in [("추천", rated[rated.grade == "추천"], False),
                                ("비추천", rated[rated.grade == "비추천"], True),
                                ("전체_한국", res[res.country == "KR"], False),
                                ("전체_미국", res[res.country == "US"], False)]:
            _table(by_group(part, asc)).to_excel(xw, sheet_name=name, index=False)
        pd.DataFrame({"설명": [
            f"종합점수 = 기술점수(50) + 가치점수(50). {BUY_SCORE}점 이상 추천, {SELL_SCORE}점 이하 비추천.",
            "기술: 이동평균 추세 20 · 3개월 모멘텀 15 · RSI 10 · 거래량 5",
            "가치(같은 업종 내 비교): PER 15 · PBR 10 · ROE 15 · 배당 5",
            "업종대분류: 한국(KSIC)·미국(Nasdaq) 업종을 같은 기준으로 묶은 것",
            "규모: 한국은 시가총액 1~100위 대형주, 101~300위 중형주 / 미국은 100억$ 이상 대형주, 20억$ 이상 중형주",
            "스타일: " + ", ".join(STYLES),
            "거래대금이 너무 적거나 상장 60거래일 미만인 종목은 '제외'.",
            "이 결과는 정해진 규칙에 따른 참고 자료이며 투자 권유가 아닙니다.",
        ]}).to_excel(xw, sheet_name="설명", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2" if ws.title != "설명" else None
            if ws.title not in ("설명", "요약"):
                ws.auto_filter.ref = ws.dimensions          # 엑셀에서 열마다 필터·정렬 버튼
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(8, width * 1.3), 60)
    return path


def write_csv(res: pd.DataFrame, as_of: str):
    path = RESULTS_DIR / f"추천결과_{as_of}.csv"
    _table(res).to_csv(path, index=False, encoding="utf-8-sig")   # 엑셀에서 한글 깨짐 방지
    return path


def _usdkrw():
    try:
        import yfinance as yf
        v = float(yf.Ticker("KRW=X").history(period="5d")["Close"].dropna().iloc[-1])
        return v if 500 < v < 3000 else 1400.0
    except Exception:
        return 1400.0


def dashboard_payload(res: pd.DataFrame, as_of: str) -> str:
    """대시보드용 데이터. 반복되는 글자(업종 등)는 번호로 바꿔 파일 크기를 줄입니다."""
    d = res[res["grade"] != "제외"].copy()
    rate = _usdkrw()
    d["cap_usd"] = (d["market_cap"] / rate).where(d["country"] == "KR", d["market_cap"]) / 1e6   # 백만 달러
    dicts = {"market": sorted(d["market"].dropna().unique()), "sector": sorted(d["sector"].fillna("").unique()),
             "group": GROUPS, "size": ["대형주", "중형주", "소형주"], "grade": ["추천", "중립", "비추천"]}
    enc = {k: {v: i for i, v in enumerate(vals)} for k, vals in dicts.items()}
    cols = ["ticker", "name", "country", "market", "sector", "group", "size", "grade", "score", "tech_score",
            "value_score", "price", "ret_1w", "ret_3m", "rsi", "per", "pbr", "roe", "div_yield", "cap_usd",
            "styles", "reasons"]
    out = d[cols].copy()
    out["sector"] = out["sector"].fillna("")
    for k in enc:
        out[k] = out[k].map(enc[k])
    for c in ("score", "tech_score", "value_score", "ret_1w", "ret_3m", "rsi", "per", "pbr", "roe", "div_yield"):
        out[c] = out[c].round(1)
    out["price"] = out["price"].round(2)
    out["cap_usd"] = out["cap_usd"].round(0)
    out["styles"] = out["styles"].fillna("").map(lambda s: [STYLES.index(x) for x in s.split("·") if x])
    payload = {"asOf": as_of, "buy": BUY_SCORE, "sell": SELL_SCORE, "usdkrw": round(rate, 1),
               "cols": cols, "dict": dicts, "styles": STYLES, "marketKo": MARKET_KO,
               "rows": json.loads(out.to_json(orient="values", force_ascii=False))}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_dashboard(res: pd.DataFrame, as_of: str):
    DOCS_DIR.mkdir(exist_ok=True)
    tpl = Path(__file__).with_name("dashboard_template.html").read_text(encoding="utf-8")
    body = tpl.replace("__PAYLOAD__", dashboard_payload(res, as_of).replace("</", "<\\/"))
    # GitHub 웹페이지용 완전한 HTML 문서 (한글 깨짐·화면 배율 문제 방지)
    html = ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
    path = DOCS_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    return path

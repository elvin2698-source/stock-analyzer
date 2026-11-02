"""분석 결과를 엑셀(results/) + 웹 대시보드(docs/index.html) + CSV로 저장합니다."""
import json
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, DOCS_DIR, BUY_SCORE, SELL_SCORE

COLS = {"ticker": "종목코드", "name": "기업명", "country": "국가", "market": "시장", "sector": "업종",
        "grade": "등급", "score": "종합점수", "tech_score": "기술점수", "value_score": "가치점수",
        "price": "현재가", "ret_1w": "1주수익률(%)", "ret_3m": "3개월수익률(%)", "rsi": "RSI",
        "per": "PER", "pbr": "PBR", "roe": "ROE(%)", "div_yield": "배당수익률(%)",
        "market_cap": "시가총액", "reasons": "판단 근거"}


def _table(df):
    out = df[[c for c in COLS if c in df]].rename(columns=COLS)
    for c in ("1주수익률(%)", "3개월수익률(%)", "RSI", "PER", "PBR", "ROE(%)", "배당수익률(%)"):
        if c in out:
            out[c] = out[c].round(2)
    return out


def write_excel(res: pd.DataFrame, as_of: str):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"추천결과_{as_of}.xlsx"
    rated = res[res["grade"] != "제외"]
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        summary = res.groupby(["country", "grade"]).size().unstack(fill_value=0)
        summary.index = summary.index.map({"KR": "한국", "US": "미국"})
        summary.to_excel(xw, sheet_name="요약")
        for name, part in [("추천", rated[rated.grade == "추천"]), ("비추천", rated[rated.grade == "비추천"]),
                           ("전체_한국", res[res.country == "KR"]), ("전체_미국", res[res.country == "US"])]:
            _table(part.sort_values("score", ascending=name == "비추천")).to_excel(xw, sheet_name=name, index=False)
        pd.DataFrame({"설명": [
            f"종합점수 = 기술점수(50) + 가치점수(50). {BUY_SCORE}점 이상 추천, {SELL_SCORE}점 이하 비추천.",
            "기술: 이동평균 추세 20 · 3개월 모멘텀 15 · RSI 10 · 거래량 5",
            "가치(같은 업종 내 비교): PER 15 · PBR 10 · ROE 15 · 배당 5",
            "거래대금이 너무 적거나 상장 60거래일 미만인 종목은 '제외'.",
            "이 결과는 정해진 규칙에 따른 참고 자료이며 투자 권유가 아닙니다.",
        ]}).to_excel(xw, sheet_name="설명", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2" if ws.title != "설명" else None
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(8, width * 1.3), 60)
    return path


def write_csv(res: pd.DataFrame, as_of: str):
    path = RESULTS_DIR / f"추천결과_{as_of}.csv"
    _table(res).to_csv(path, index=False, encoding="utf-8-sig")   # 엑셀에서 한글 깨짐 방지
    return path


def write_dashboard(res: pd.DataFrame, as_of: str):
    DOCS_DIR.mkdir(exist_ok=True)
    keep = ["ticker", "name", "country", "sector", "grade", "score", "tech_score", "value_score",
            "price", "ret_1w", "ret_3m", "rsi", "per", "pbr", "roe", "div_yield", "reasons"]
    data = res[res["grade"] != "제외"][keep].round(2)
    rows = json.loads(data.to_json(orient="values", force_ascii=False))
    tpl = (Path(__file__).with_name("dashboard_template.html")).read_text(encoding="utf-8")
    html = (tpl.replace("__AS_OF__", as_of)
               .replace("__BUY__", str(BUY_SCORE)).replace("__SELL__", str(SELL_SCORE))
               .replace("__DATA__", json.dumps(rows, ensure_ascii=False, separators=(",", ":"))))
    path = DOCS_DIR / "index.html"
    path.write_text(html, encoding="utf-8")
    return path

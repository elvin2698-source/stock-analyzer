"""종목 분류: 업종 대분류(한국·미국 공통), 규모, 투자 스타일.

한국 업종(KSIC 160여 개)과 미국 업종(Nasdaq 분류 150여 개)을 같은 대분류로 묶어서
두 나라 종목을 한 기준으로 비교할 수 있게 합니다. 위에서부터 먼저 맞는 규칙이 적용됩니다.
"""
import numpy as np
import pandas as pd

GROUPS = ["반도체·전자부품", "소프트웨어·인터넷", "IT하드웨어·통신장비", "통신서비스", "미디어·엔터·레저",
          "바이오·제약", "의료기기·헬스케어", "2차전지·전기장비", "자동차·부품", "기계·조선·방산·항공",
          "화학·소재", "철강·금속·광업", "에너지·유틸리티", "건설·부동산", "금융", "식음료·농수산",
          "패션·의류", "생활소비재", "유통·도소매", "운송·물류·여행", "교육", "지주회사", "기타 서비스"]

KR_RULES = [
    ("반도체·전자부품", ["반도체", "전자부품", "마그네틱"]),
    ("소프트웨어·인터넷", ["소프트웨어", "컴퓨터 프로그래밍", "정보 서비스", "자료처리", "포털"]),
    ("IT하드웨어·통신장비", ["통신 및 방송 장비", "컴퓨터 및 주변", "영상 및 음향기기", "사진장비",
                          "측정, 시험", "정보통신장비"]),
    ("통신서비스", ["전기 통신", "통신 공사"]),
    ("미디어·엔터·레저", ["영화", "방송", "오디오물", "출판", "광고", "창작", "유원지", "영상·오디오",
                       "스포츠", "기록매체", "인쇄"]),
    ("바이오·제약", ["의약", "생물", "자연과학 및 공학"]),
    ("의료기기·헬스케어", ["의료", "보건"]),
    ("2차전지·전기장비", ["전지", "전동기", "전기장비", "절연선", "전구"]),
    ("자동차·부품", ["자동차"]),
    ("기계·조선·방산·항공", ["기계", "선박", "항공기", "무기", "운송장비 제조"]),
    ("화학·소재", ["화학", "플라스틱", "고무", "비료", "유리", "시멘트", "요업", "비금속", "펄프", "종이",
                 "골판지", "목재", "나무제품"]),
    ("철강·금속·광업", ["철강", "금속", "비철", "주조", "광업"]),
    ("에너지·유틸리티", ["석유", "가스", "전기업", "증기", "연료", "폐기물", "원료 재생"]),
    ("건설·부동산", ["건설", "건물", "토목", "공사", "건축", "부동산", "시설물"]),
    ("금융", ["은행", "보험", "금융", "신탁", "연금"]),
    ("식음료·농수산", ["식품", "식료품", "음료", "육류", "수산물", "곡물", "유지", "과실", "담배", "떡",
                    "도시락", "사료", "작물", "어로", "농·축산물"]),
    ("패션·의류", ["의복", "섬유", "가죽", "신발", "직물", "방적", "편조", "귀금속"]),
    ("운송·물류·여행", ["운송", "여행", "숙박", "창고"]),
    ("유통·도소매", ["소매", "도매", "상품 중개", "판매업"]),
    ("교육", ["교육", "교습", "학원"]),
    ("생활소비재", ["가구", "가정용 기기", "악기", "운동 및 경기", "그외 기타 제품"]),
    ("지주회사", ["회사 본부"]),
]

US_RULES = [
    ("반도체·전자부품", ["semiconductor", "electronic components", "electronics distribution"]),
    ("소프트웨어·인터넷", ["software", "edp services", "programming data"]),
    ("IT하드웨어·통신장비", ["computer manufacturing", "computer peripheral", "computer communications",
                          "telecommunications equipment", "communications equipment", "consumer electronics/appliances",
                          "office equipment"]),
    ("통신서비스", ["cable & other pay"]),
    ("미디어·엔터·레저", ["movies", "broadcasting", "publishing", "newspapers", "books", "advertising",
                       "amusement", "recreational games"]),
    ("의료기기·헬스케어", ["medical", "hospital", "ophthalmic", "electromedical", "managed health",
                        "laboratory analytical", "precision instruments"]),
    ("바이오·제약", ["biotechnology", "pharmaceutical", "medicinal", "biological"]),
    ("2차전지·전기장비", ["electrical products"]),
    ("자동차·부품", ["auto", "motor vehicles"]),
    ("기계·조선·방산·항공", ["aerospace", "military", "ordnance", "machinery", "construction/ag equipment",
                          "fluid controls", "pollution control", "tools/hardware"]),
    ("에너지·유틸리티", ["oil", "gas", "coal", "electric utilities", "power generation", "water supply",
                      "environmental services"]),
    ("화학·소재", ["chemical", "plastic", "paints", "containers/packaging", "paper", "forest products"]),
    ("철강·금속·광업", ["steel", "metal", "aluminum", "mining"]),
    ("건설·부동산", ["real estate", "homebuilding", "building", "construction", "water sewer", "general bldg"]),
    ("금융", ["bank", "insur", "finance", "investment", "savings", "trusts", "blank checks", "financial"]),
    ("식음료·농수산", ["food", "beverages", "farming", "meat", "tobacco", "restaurants"]),
    ("패션·의류", ["apparel", "clothing", "shoe", "textiles", "garments"]),
    ("운송·물류·여행", ["marine transportation", "trucking", "air freight", "railroads", "transportation",
                     "logistics", "hotels"]),
    ("유통·도소매", ["retail", "stores", "catalog", "wholesale", "distributors", "chains"]),
    ("생활소비재", ["cosmetics", "home furnishings", "durable goods", "consumer specialties"]),
]
US_SECTOR_FALLBACK = {
    "Health Care": "바이오·제약", "Technology": "소프트웨어·인터넷", "Finance": "금융",
    "Energy": "에너지·유틸리티", "Utilities": "에너지·유틸리티", "Real Estate": "건설·부동산",
    "Basic Materials": "화학·소재", "Consumer Staples": "식음료·농수산",
    "Telecommunications": "통신서비스", "Industrials": "기계·조선·방산·항공",
}


def _match(text, rules):
    for group, words in rules:
        if any(w in text for w in words):
            return group
    return None


def sector_group(df: pd.DataFrame, market: str) -> pd.Series:
    if market == "KR":
        return df["sector"].fillna("").map(lambda s: _match(s, KR_RULES) or "기타 서비스")
    ind = df.get("industry", pd.Series("", index=df.index)).fillna("").str.lower()
    sec = df["sector"].fillna("")
    return pd.Series([_match(i, US_RULES) or US_SECTOR_FALLBACK.get(s, "기타 서비스")
                      for i, s in zip(ind, sec)], index=df.index)


def size_class(df: pd.DataFrame, market: str) -> pd.Series:
    cap = pd.to_numeric(df["market_cap"], errors="coerce")
    if market == "KR":   # 거래소 기준과 같게: 시가총액 1~100위 대형, 101~300위 중형, 나머지 소형
        rank = cap.rank(ascending=False, method="first")
        return pd.Series(np.select([rank <= 100, rank <= 300], ["대형주", "중형주"], "소형주"), index=df.index)
    return pd.Series(np.select([cap >= 10e9, cap >= 2e9], ["대형주", "중형주"], "소형주"), index=df.index)


STYLES = ["저평가", "고수익성", "고배당", "상승추세", "하락추세", "모멘텀 강세", "과매도", "과열", "적자"]


def styles(df: pd.DataFrame, per_pts: pd.Series) -> pd.Series:
    rules = {
        "저평가": (per_pts >= 10.5) & df["per"].notna() | df["pbr"].between(0, 1, inclusive="neither"),
        "고수익성": df["roe"] >= 15,
        "고배당": df["div_yield"] >= 4,
        "상승추세": (df["price"] > df["ma200"]) & (df["ma20"] > df["ma60"]),
        "하락추세": (df["price"] < df["ma200"]) & (df["ma20"] < df["ma60"]),
        "모멘텀 강세": df["ret_3m"].rank(pct=True) >= 0.8,
        "과매도": df["rsi"] < 30,
        "과열": df["rsi"] > 70,
        "적자": df["roe"] <= 0,
    }
    flags = pd.DataFrame({k: v.fillna(False).astype(bool) for k, v in rules.items()})
    return flags.apply(lambda r: "·".join(k for k in STYLES if r[k]), axis=1)

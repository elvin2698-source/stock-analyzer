"""전체 실행: 데이터 수집 → 분석 → 결과 저장

사용법:
  python src/main.py            # 한국 + 미국
  python src/main.py --only KR  # 한국만
  python src/main.py --no-collect  # 이미 받은 CSV로 분석만
"""
import argparse
import sys

import pandas as pd

from config import DATA_DIR
import analyze
import report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["KR", "US"])
    ap.add_argument("--no-collect", action="store_true")
    args = ap.parse_args()
    markets = [args.only] if args.only else ["KR", "US"]
    DATA_DIR.mkdir(exist_ok=True)

    results, failed = [], []
    for m in markets:
        try:
            if not args.no_collect:
                (__import__("collect_kr") if m == "KR" else __import__("collect_us")).run()
            low = m.lower()
            prices = pd.read_csv(DATA_DIR / f"prices_{low}.csv", dtype={"ticker": str})
            comp = pd.read_csv(DATA_DIR / f"companies_{low}.csv", dtype={"ticker": str})
            res = analyze.analyze(prices, comp, m)
            results.append(res)
            print(f"[{m}] 분석 완료: " + ", ".join(f"{g} {n}" for g, n in res["grade"].value_counts().items()))
        except Exception as e:
            failed.append(m)
            print(f"[{m}] 실패: {type(e).__name__}: {e}", file=sys.stderr)

    if not results:
        sys.exit("분석할 데이터가 없습니다.")
    res = pd.concat(results, ignore_index=True)
    as_of = max(r["as_of"].iloc[0] for r in results)
    for p in (report.write_excel(res, as_of), report.write_csv(res, as_of), report.write_dashboard(res, as_of)):
        print("저장:", p)
    if failed:
        sys.exit(f"일부 시장 실패: {failed}")


if __name__ == "__main__":
    main()

"""전체 실행: 데이터 수집 → 분석 → 결과 저장

사용법:
  python src/main.py            # 한국 + 미국
  python src/main.py --only KR  # 한국만
  python src/main.py --no-collect  # 이미 받은 CSV로 분석만
"""
import argparse
import sys

import pandas as pd

from config import DATA_DIR, read_csv
import analyze
import fundamentals
import paper
import report
import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["KR", "US"])
    ap.add_argument("--no-collect", action="store_true")
    args = ap.parse_args()
    markets = [args.only] if args.only else ["KR", "US"]
    DATA_DIR.mkdir(exist_ok=True)

    results, failed, prices_by = {}, [], {}
    usdkrw = report.usdkrw()
    for m in markets:
        try:
            if not args.no_collect:
                (__import__("collect_kr") if m == "KR" else __import__("collect_us")).run()
                try:   # 백테스트용 과거 재무 (실패해도 분석은 계속)
                    c0 = read_csv(DATA_DIR / f"companies_{m.lower()}.csv").sort_values("market_cap", ascending=False)
                    fundamentals.update(m, c0["ticker"].tolist())
                except Exception as e:
                    print(f"[{m}] 재무 이력 받기 실패: {type(e).__name__}: {e}", file=sys.stderr)
            low = m.lower()
            prices = read_csv(DATA_DIR / f"prices_{low}.v2.csv.gz")
            comp = read_csv(DATA_DIR / f"companies_{low}.csv")
            res = analyze.analyze(prices, comp, m)
            results[m] = res
            prices_by[m] = prices
            print(f"[{m}] 분석 완료: " + ", ".join(f"{g} {n}" for g, n in res["grade"].value_counts().items()))
        except Exception as e:
            failed.append(m)
            print(f"[{m}] 실패: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        try:
            validate.save_signals(m, res)
            paper.run(m, res, prices, usdkrw)
        except Exception as e:
            failed.append(f"모의투자-{m}")
            print(f"[{m}] 모의투자 실패: {type(e).__name__}: {e}", file=sys.stderr)

    if not results:
        sys.exit("분석할 데이터가 없습니다.")
    res = pd.concat(results.values(), ignore_index=True)
    as_of = max(r["as_of"].iloc[0] for r in results.values())
    paper_data = paper.load_for_dashboard(results, usdkrw)
    paper_data.update(validate.run_all(prices_by, results))
    try:
        paper_data["weekly"] = paper.weekly_report()
    except Exception as e:
        print(f"주간 리포트 실패: {type(e).__name__}: {e}", file=sys.stderr)
    for p in (report.write_excel(res, as_of), report.write_csv(res, as_of),
              report.write_dashboard(res, as_of, paper_data, usdkrw)):
        print("저장:", p)
    if failed:
        sys.exit(f"일부 시장 실패: {failed}")


if __name__ == "__main__":
    main()

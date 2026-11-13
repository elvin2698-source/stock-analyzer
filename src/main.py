"""전체 실행: 데이터 수집 → 분석 → 모의투자 → 검증 → 결과 저장

한 단계가 실패해도 전체가 멈추지 않게 만들었습니다.
- 수집 실패   → 전에 받아 둔 데이터로 분석 계속 (상태: 이전 데이터 사용)
- 분석 실패   → 마지막으로 성공한 분석 결과로 대시보드를 계속 만듦 (상태: 이전 분석 사용)
- 모의투자·검증·엑셀 실패 → 그 부분만 건너뜀
- 대시보드는 두 시장 중 하나라도 결과가 있으면 항상 새로 만듦
실행 상태는 docs/status.json 에 남기고 대시보드 맨 위에 경고로 보여줍니다.
종료 코드: 대시보드를 새로 만들었으면 0(일부 실패가 있어도), 아무것도 못 만들었으면 1

사용법:
  python src/main.py            # 한국 + 미국
  python src/main.py --only KR  # 한국만
  python src/main.py --no-collect  # 이미 받은 데이터로 분석만
"""
import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import pandas as pd

from config import DATA_DIR, DOCS_DIR, read_csv
import analyze
import fundamentals
import paper
import report
import validate

STATUS_FILE = DOCS_DIR / "status.json"
SAFE_MODE = os.environ.get("SAFE_MODE") == "1"     # 자동 검사가 실패한 날: 모의투자는 쉼
STAGE_KO = {"collect": "데이터 수집", "fundamentals": "과거 재무", "analyze": "분석", "paper": "모의투자",
            "signals": "점수 기록", "validate": "검증·백테스트", "weekly": "주간 리포트", "excel": "엑셀·CSV",
            "dashboard": "대시보드"}


class Status:
    def __init__(self):
        self.markets, self.stages, self.errors, self.as_of = {}, {}, [], {}

    def mark(self, where, stage, state, err=None):
        """state: ok / fallback(이전 데이터로 대신) / skip / fail"""
        (self.markets.setdefault(where, {}) if where != "전체" else self.stages)[stage] = state
        if err is not None:
            msg = f"{where} {STAGE_KO.get(stage, stage)}: {type(err).__name__}: {str(err)[:300]}"
            self.errors.append(msg)
            print("[실패] " + msg, file=sys.stderr)
            traceback.print_exception(err, file=sys.stderr, limit=4)

    def level(self):
        states = [s for m in self.markets.values() for s in m.values()] + list(self.stages.values())
        if self.stages.get("dashboard") == "fail" or not states:
            return "failed"
        return "ok" if all(s in ("ok", "skip") for s in states) else "degraded"


def last_analysis_path(m):
    return DATA_DIR / f"last_analysis_{m.lower()}.csv.gz"


def collect_market(m):
    (__import__("collect_kr") if m == "KR" else __import__("collect_us")).run()


def run_market(m, collect, usdkrw, status: Status):
    """한 시장 처리. 돌려주는 값: (분석 결과, 주가, 새로 분석했는지)"""
    low = m.lower()
    if collect:
        try:
            collect_market(m)
            status.mark(m, "collect", "ok")
        except Exception as e:
            status.mark(m, "collect", "fallback", e)          # 아래에서 전에 받아 둔 파일로 계속
        try:
            c0 = read_csv(DATA_DIR / f"companies_{low}.csv").sort_values("market_cap", ascending=False)
            fundamentals.update(m, c0["ticker"].tolist())
            status.mark(m, "fundamentals", "ok")
        except Exception as e:
            status.mark(m, "fundamentals", "fallback", e)

    prices = None
    try:
        prices = read_csv(DATA_DIR / f"prices_{low}.v2.csv.gz")
        comp = read_csv(DATA_DIR / f"companies_{low}.csv")
        res = analyze.analyze(prices, comp, m)
        res.to_csv(last_analysis_path(m), index=False, compression="gzip")   # 다음에 실패하면 대신 쓸 결과
        status.mark(m, "analyze", "ok")
        fresh = True
        print(f"[{m}] 분석 완료: " + ", ".join(f"{g} {n}" for g, n in res["grade"].value_counts().items()))
    except Exception as e:
        p = last_analysis_path(m)
        if not p.exists():
            status.mark(m, "analyze", "fail", e)
            return None, None, False
        res = read_csv(p)
        res["styles"] = res["styles"].fillna("")
        status.mark(m, "analyze", "fallback", e)
        fresh = False
        print(f"[{m}] 분석 실패 → 마지막 분석 결과({res['as_of'].iloc[0]})로 대신 표시", file=sys.stderr)
    status.as_of[m] = str(res["as_of"].iloc[0])

    if fresh and SAFE_MODE:
        status.mark(m, "paper", "fallback", RuntimeError("자동 검사가 실패해서 오늘은 모의투자를 쉬어요 (기록 보호)"))
        fresh_paper = False
    else:
        fresh_paper = fresh
    if fresh_paper:                                       # 옛 결과로는 모의투자·점수 기록을 하지 않음
        for stage, fn in (("signals", lambda: validate.save_signals(m, res)),
                          ("paper", lambda: paper.run(m, res, prices, usdkrw))):
            try:
                fn()
                status.mark(m, stage, "ok")
            except Exception as e:
                status.mark(m, stage, "fail", e)
    elif not fresh:
        status.mark(m, "paper", "skip")
    return res, (prices if fresh else None), fresh


def write_status(status: Status):
    prev = {}
    if STATUS_FILE.exists():
        try:
            prev = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    level = status.level()
    out = {"run_at": now, "level": level, "markets": status.markets, "stages": status.stages, "as_of": status.as_of,
           "stage_names": STAGE_KO,
           "errors": status.errors[:20],
           "last_ok": now if level == "ok" else prev.get("last_ok"),
           "last_update": now if level != "failed" else prev.get("last_update")}
    DOCS_DIR.mkdir(exist_ok=True)
    STATUS_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["KR", "US"])
    ap.add_argument("--no-collect", action="store_true")
    args = ap.parse_args(argv)
    markets = [args.only] if args.only else ["KR", "US"]
    DATA_DIR.mkdir(exist_ok=True)
    status = Status()

    try:
        usdkrw = report.usdkrw()
    except Exception:
        usdkrw = 1400.0
    results, prices_by = {}, {}
    for m in markets:
        res, prices, fresh = run_market(m, not args.no_collect, usdkrw, status)
        if res is not None:
            results[m] = res
        if prices is not None:
            prices_by[m] = prices

    if not results:
        status.mark("전체", "dashboard", "fail", RuntimeError("두 시장 모두 결과가 없어 대시보드를 그대로 둠"))
        st = write_status(status)
        print("분석할 데이터가 없어 대시보드를 새로 만들지 못했어요.", file=sys.stderr)
        return 1

    res = pd.concat(results.values(), ignore_index=True)
    as_of = max(str(r["as_of"].iloc[0]) for r in results.values())
    paper_data = {}
    for stage, fn in (("paper", lambda: paper_data.update(paper.load_for_dashboard(results, usdkrw))),
                      ("validate", lambda: paper_data.update(validate.run_all(prices_by, results))),
                      ("weekly", lambda: paper_data.__setitem__("weekly", paper.weekly_report()))):
        try:
            fn()
            if stage != "paper":
                status.mark("전체", stage, "ok")
        except Exception as e:
            status.mark("전체", stage, "fail", e)
    try:
        for p in (report.write_excel(res, as_of), report.write_csv(res, as_of)):
            print("저장:", p)
        status.mark("전체", "excel", "ok")
    except Exception as e:
        status.mark("전체", "excel", "fail", e)

    status.mark("전체", "dashboard", "ok")          # 상태를 먼저 정해야 대시보드에 넣을 수 있음
    st = write_status(status)
    try:
        print("저장:", report.write_dashboard(res, as_of, paper_data, usdkrw, status=st))
    except Exception as e:
        status.mark("전체", "dashboard", "fail", e)
        write_status(status)
        return 1
    print(f"상태: {st['level']}" + ("" if not st["errors"] else " · " + " / ".join(st["errors"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())

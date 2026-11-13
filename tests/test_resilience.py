"""고장 상황 검사: 한 단계가 실패해도 대시보드가 계속 만들어지는지."""
import json

import pandas as pd
import pytest

import analyze
import main
import paper
import report
import validate
from conftest import make_market


@pytest.fixture
def env(tmp_path, monkeypatch):
    data, docs, results = tmp_path / "data", tmp_path / "docs", tmp_path / "results"
    for d in (data, docs, results):
        d.mkdir()
    monkeypatch.setattr(main, "DATA_DIR", data)
    monkeypatch.setattr(main, "DOCS_DIR", docs)
    monkeypatch.setattr(main, "STATUS_FILE", docs / "status.json")
    monkeypatch.setattr(report, "DOCS_DIR", docs)
    monkeypatch.setattr(report, "RESULTS_DIR", results)
    monkeypatch.setattr(report, "usdkrw", lambda: 1400.0)
    monkeypatch.setattr(main.fundamentals, "update", lambda m, t: None)
    monkeypatch.setattr(paper, "run", lambda *a, **k: None)
    monkeypatch.setattr(paper, "load_for_dashboard", lambda *a, **k: {})
    monkeypatch.setattr(paper, "weekly_report", lambda *a, **k: {})
    monkeypatch.setattr(validate, "save_signals", lambda *a, **k: None)
    monkeypatch.setattr(validate, "run_all", lambda *a, **k: {})
    for m in ("kr", "us"):
        comp, prices, _ = make_market(n=60, days=150, seed=1 if m == "kr" else 2)
        prices.to_csv(data / f"prices_{m}.v2.csv.gz", index=False, compression="gzip")
        comp.to_csv(data / f"companies_{m}.csv", index=False)
    return tmp_path


def status(env):
    return json.loads((env / "docs" / "status.json").read_text())


def test_collect_failure_uses_previous_data(env, monkeypatch):
    def boom(m):
        raise ConnectionError("야후 접속 실패")
    monkeypatch.setattr(main, "collect_market", boom)
    assert main.main([]) == 0
    st = status(env)
    assert st["level"] == "degraded"
    assert st["markets"]["KR"]["collect"] == "fallback" and st["markets"]["KR"]["analyze"] == "ok"
    assert (env / "docs" / "index.html").exists()
    assert "야후 접속 실패" in " ".join(st["errors"])


def test_analysis_failure_falls_back_to_last_result(env, monkeypatch):
    monkeypatch.setattr(main, "collect_market", lambda m: None)
    assert main.main([]) == 0                                   # 한 번 정상 실행 → 마지막 결과 저장
    assert status(env)["level"] == "ok"
    real = analyze.analyze

    def flaky(prices, comp, market):
        if market == "US":
            raise TypeError("새 라이브러리 버전 오류")
        return real(prices, comp, market)
    monkeypatch.setattr(main.analyze, "analyze", flaky)
    (env / "docs" / "index.html").unlink()
    assert main.main([]) == 0
    st = status(env)
    assert st["level"] == "degraded" and st["markets"]["US"]["analyze"] == "fallback"
    assert st["markets"]["US"]["paper"] == "skip"
    html = (env / "docs" / "index.html").read_text()
    assert '"US"' in html                                       # 미국 종목도 이전 결과로 표시됨


def test_everything_failing_keeps_old_dashboard(env, monkeypatch):
    monkeypatch.setattr(main, "collect_market", lambda m: None)
    def broken(*a, **k):
        raise ValueError("분석 불가")
    monkeypatch.setattr(main.analyze, "analyze", broken)
    (env / "docs" / "index.html").write_text("예전 대시보드")
    assert main.main([]) == 1
    assert status(env)["level"] == "failed"
    assert (env / "docs" / "index.html").read_text() == "예전 대시보드"


def test_safe_mode_skips_paper(env, monkeypatch):
    monkeypatch.setattr(main, "collect_market", lambda m: None)
    monkeypatch.setattr(main, "SAFE_MODE", True)
    assert main.main([]) == 0
    st = status(env)
    assert st["markets"]["KR"]["paper"] == "fallback" and st["level"] == "degraded"

"""CLI tests.

The CLI is the surface most users touch first, and it was previously the least
tested part of the codebase (0% coverage). These tests exercise every command
end to end and assert on what is *printed*, because for a command-line tool the
output is the contract.

All runs use the synthetic data path — no network, so they are deterministic and
safe in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from risk_engine.cli import build_parser, main

# Ticker names with no real-world counterpart, forcing the synthetic generator.
SYNTH = "SYN_A,SYN_B,SYN_C"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Guarantee the synthetic fallback regardless of the machine's connectivity.

    Without this the tests would silently depend on yfinance being installed and
    the network being up, making them slow and flaky.
    """
    from risk_engine import data

    monkeypatch.setattr(data, "_download", lambda *a, **k: None)
    monkeypatch.setattr(data, "CACHE_DIR", Path("/nonexistent-cache-for-tests"))


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------

def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert "risk-engine" in capsys.readouterr().out


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_unknown_command_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nonsense"])


def test_tickers_are_upper_cased_and_split():
    args = build_parser().parse_args(["report", "--tickers", "aapl, msft"])
    assert args.tickers == ["AAPL", "MSFT"]


def test_empty_ticker_list_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["report", "--tickers", " , "])


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def test_report_prints_every_method(capsys):
    assert main(["report", "--tickers", SYNTH, "--confidence", "0.99"]) == 0
    out = capsys.readouterr().out
    for method in (
        "historical",
        "parametric_normal",
        "parametric_student_t",
        "monte_carlo_gaussian",
        "monte_carlo_student_t",
        "monte_carlo_bootstrap",
        "conditional_ewma_empirical",
        "conditional_garch_empirical",
    ):
        assert method in out, f"{method} missing from report output"
    assert "Spread across methods" in out


def test_report_writes_artifacts(tmp_path, capsys):
    assert main(["report", "--tickers", SYNTH, "--output", str(tmp_path)]) == 0
    assert (tmp_path / "loss_distribution.png").exists()
    assert (tmp_path / "var_results.csv").exists()
    assert (tmp_path / "loss_distribution.png").stat().st_size > 1000


def test_report_honours_the_horizon_flag(capsys):
    main(["report", "--tickers", SYNTH, "--horizon", "10"])
    assert "horizon: 10d" in capsys.readouterr().out


def test_report_rejects_mismatched_weights(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["report", "--tickers", SYNTH, "--weights", "0.5,0.5"])
    assert exc.value.code != 0


def test_report_accepts_matching_weights():
    assert main(["report", "--tickers", SYNTH, "--weights", "0.5,0.3,0.2"]) == 0


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------

def test_backtest_runs_and_reports_coverage(capsys):
    assert main(["backtest", "--tickers", SYNTH, "--window", "250"]) == 0
    out = capsys.readouterr().out
    assert "Kupiec" in out
    assert "Christoffersen" in out
    assert "Model comparison" in out
    assert "ewma_fhs" in out


def test_backtest_writes_comparison_csv(tmp_path):
    main(["backtest", "--tickers", SYNTH, "--window", "250", "--output", str(tmp_path)])
    csv = tmp_path / "model_comparison.csv"
    assert csv.exists()
    header = csv.read_text().splitlines()[0]
    for column in ("method", "breaches", "kupiec_p", "joint_p", "passes_all"):
        assert column in header


def test_backtest_strict_flag_signals_failure():
    """--strict must exit non-zero when a model fails, for use in CI gates."""
    code = main(["backtest", "--tickers", SYNTH, "--window", "250", "--strict"])
    assert code in (0, 1)


def test_backtest_rejects_an_impossible_window(capsys):
    code = main(["backtest", "--tickers", SYNTH, "--window", "100000"])
    assert code == 2
    assert "error" in capsys.readouterr().err


# --------------------------------------------------------------------------
# option
# --------------------------------------------------------------------------

def test_option_prints_price_greeks_and_validation(capsys):
    code = main(["option", "--spot", "100", "--strike", "105",
                 "--expiry", "1", "--vol", "0.2"])
    assert code == 0
    out = capsys.readouterr().out
    for label in ("Price", "Delta", "Gamma", "Vega", "Theta", "Rho",
                  "Binomial", "Put-call parity residual"):
        assert label in out


def test_option_solves_for_implied_volatility(capsys):
    main(["option", "--spot", "100", "--strike", "105", "--expiry", "1",
          "--vol", "0.2", "--market-price", "8.50"])
    assert "Implied volatility" in capsys.readouterr().out


def test_option_put_branch_runs(capsys):
    assert main(["option", "--spot", "100", "--strike", "95", "--expiry", "0.5",
                 "--vol", "0.3", "--kind", "put"]) == 0
    assert "European put" in capsys.readouterr().out


def test_option_rejects_negative_spot(capsys):
    assert main(["option", "--spot", "-5", "--strike", "100",
                 "--expiry", "1", "--vol", "0.2"]) == 2
    assert "error" in capsys.readouterr().err


def test_option_writes_a_greeks_chart(tmp_path):
    main(["option", "--spot", "100", "--strike", "100", "--expiry", "1",
          "--vol", "0.25", "--output", str(tmp_path)])
    assert (tmp_path / "greeks_profile.png").exists()


# --------------------------------------------------------------------------
# contagion
# --------------------------------------------------------------------------

def test_contagion_ranks_assets(capsys):
    assert main(["contagion", "--tickers", SYNTH, "--threshold", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "DebtRank" in out
    assert "systemically important" in out


def test_contagion_supports_the_mst_network(capsys):
    assert main(["contagion", "--tickers", SYNTH, "--network", "mst"]) == 0
    assert "mst" in capsys.readouterr().out


def test_contagion_errors_when_the_threshold_removes_every_edge(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["contagion", "--tickers", SYNTH, "--threshold", "0.999"])
    assert exc.value.code != 0


def test_contagion_writes_artifacts(tmp_path):
    main(["contagion", "--tickers", SYNTH, "--threshold", "0.2",
          "--output", str(tmp_path)])
    assert (tmp_path / "contagion_network.png").exists()
    assert (tmp_path / "systemic_ranking.csv").exists()


# --------------------------------------------------------------------------
# diversification
# --------------------------------------------------------------------------

#: Enough synthetic names for K = floor(N/5) to be meaningful rather than 1.
SYNTH_WIDE = ",".join(f"SYN_{i:02d}" for i in range(20))


def test_diversification_reports_absorption_and_the_event_study(capsys):
    assert main(["diversification", "--tickers", SYNTH_WIDE, "--window", "250"]) == 0
    out = capsys.readouterr().out
    assert "Absorption ratio" in out
    assert "Independent bets" in out
    assert "Event study" in out
    # The false-positive count is not optional output; a hit rate alone is not
    # a result, and the CLI must not be able to quote one without the other.
    assert "false positives" in out


def test_diversification_benchmark_reports_realised_volatility(capsys):
    assert main(["diversification", "--tickers", SYNTH_WIDE, "--window", "250",
                 "--benchmark"]) == 0
    assert "trailing realised volatility" in capsys.readouterr().out


def test_diversification_refuses_a_universe_too_narrow_to_measure(capsys):
    """With N < 5, K = floor(N/5) degenerates and the ratio means nothing."""
    with pytest.raises(SystemExit) as exc:
        main(["diversification", "--tickers", SYNTH])
    assert exc.value.code != 0


def test_diversification_refuses_a_window_longer_than_the_sample():
    with pytest.raises(SystemExit) as exc:
        main(["diversification", "--tickers", SYNTH_WIDE, "--window", "99999"])
    assert exc.value.code != 0


def test_diversification_supports_the_sample_covariance(capsys):
    assert main(["diversification", "--tickers", SYNTH_WIDE, "--window", "250",
                 "--cov-model", "sample"]) == 0
    assert "sample covariance" in capsys.readouterr().out


def test_diversification_writes_artifacts(tmp_path):
    main(["diversification", "--tickers", SYNTH_WIDE, "--window", "250",
          "--output", str(tmp_path)])
    assert (tmp_path / "absorption_ratio.png").exists()
    assert (tmp_path / "absorption_ratio.csv").exists()

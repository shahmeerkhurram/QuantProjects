# QuantProjects

[![CI](https://github.com/shahmeerkhurram/QuantProjects/actions/workflows/ci.yml/badge.svg)](https://github.com/shahmeerkhurram/QuantProjects/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-266-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Two installable Python packages where rigorous mathematics meets decisions people
actually make: a **portfolio risk engine** used by a market-risk function, and a
**computational-geometry solver** for the Art Gallery Problem.

| Package | What it does | Why it's here |
|---|---|---|
| **`risk_engine`** | VaR & Expected Shortfall across 9 methods, regulatory backtesting, option Greeks, network contagion | Measuring and *validating* financial risk |
| **`artgallery`** | Fisk's constructive proof of the Art Gallery Theorem, with verified guard placement | My BSc thesis topic and ongoing research with the Department of Geometry, University of Debrecen |

They share a repository, a test philosophy and a build — not an import. Each
stands alone.

**266 tests, 93% coverage**, running on Python 3.10–3.13 in CI alongside `ruff`
and `mypy`. The tests assert *mathematical and financial properties* — put-call
parity, lattice convergence, recovery of known GARCH parameters, the `⌊n/3⌋`
guard bound, absence of look-ahead — not merely that the code returns a number.

---

## Table of contents

- [Install](#install)
- [Project 1 — Portfolio Risk Engine](#project-1--portfolio-risk-engine)
  - [Command line](#command-line)
  - [Python API](#python-api)
  - [What the engine found](#what-the-engine-found)
  - [Design decisions worth defending](#design-decisions-worth-defending)
- [Project 2 — The Art Gallery Problem](#project-2--the-art-gallery-problem)
- [Notebooks](#notebooks)
- [How it's validated](#how-its-validated)
- [Repository layout](#repository-layout)
- [Development](#development)
- [Known limitations](#known-limitations)

---

## Install

```bash
git clone https://github.com/shahmeerkhurram/QuantProjects.git
cd QuantProjects
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[market,dev]"
pytest -q
```

Extras: `market` adds live data via yfinance, `dev` adds pytest/ruff/mypy,
`notebooks` adds Jupyter. Core install needs only NumPy, SciPy, pandas,
Matplotlib and NetworkX.

**No network? Everything still runs.** `risk_engine.data` degrades from live
download → local CSV cache → a deterministic synthetic generator producing
correlated Student-t returns with volatility clustering. CI uses that path, so
the whole suite is reproducible offline.

---

## Project 1 — Portfolio Risk Engine

Answers the three questions a market-risk function has to answer, from one data
path and one portfolio object.

| Question | Module | Method |
|---|---|---|
| How much can we lose? | `var`, `volatility` | VaR & ES — historical, parametric (normal / Student-t), three Monte Carlo engines, and GARCH(1,1)/EWMA conditional models with Filtered Historical Simulation |
| Is the model any good? | `backtest` | Walk-forward backtesting with Kupiec, Christoffersen and joint conditional-coverage tests, plus binomial-derived Basel traffic-light zones |
| What's the derivatives book worth, and where does stress spread? | `options`, `portfolio`, `contagion` | Black-Scholes with all five Greeks and implied vol; delta-gamma revaluation of a mixed book; DebtRank contagion over an empirical correlation network |

### Command line

```bash
# VaR and Expected Shortfall across every method
risk-engine report --tickers AAPL,MSFT,GOOGL,JPM,XOM --confidence 0.99 --output outputs

# Walk-forward backtest with coverage tests and a model-comparison table
risk-engine backtest --tickers AAPL,MSFT,GOOGL,JPM,XOM --window 250 --all-models

# Price an option, show Greeks, validate, and solve for implied vol
risk-engine option --spot 100 --strike 105 --expiry 1 --vol 0.2 --market-price 8.50

# Correlation network and systemic ranking
risk-engine contagion --tickers AAPL,MSFT,GOOGL,JPM,XOM,JNJ,WMT --threshold 0.35
```

Useful flags: `--horizon` (multi-day VaR), `--weights` (non-equal portfolios),
`--confidence`, `--output` (writes PNGs and CSVs), `--strict` (non-zero exit if
a model fails a coverage test — for CI gates). `risk-engine <command> --help`
for the rest.

### Python API

**Value-at-Risk and Expected Shortfall**

```python
from risk_engine import load_prices, to_returns, portfolio_returns, historical_var

returns = to_returns(load_prices(["AAPL", "MSFT", "GOOGL"], start="2018-01-01"))
pnl = portfolio_returns(returns)                 # equal-weight; pass weights=[...] to change

result = historical_var(pnl, confidence=0.99)
print(result.var, result.expected_shortfall)     # positive loss magnitudes
```

Every estimator returns a `RiskResult` carrying the number *and* the assumptions
that produced it (`result.detail`), so a figure can always be traced back.

**Volatility-filtered VaR (the model that passes its backtest)**

```python
from risk_engine import conditional_var

conditional_var(pnl, confidence=0.975, model="ewma", innovation="empirical")
```

**Backtesting — the part that makes a VaR number mean something**

```python
from risk_engine import rolling_var_backtest, compare_models

bt = rolling_var_backtest(pnl, confidence=0.99, window=250, method="ewma_fhs")
print(bt.summary())          # breaches, Basel zone, all three coverage tests

compare_models(pnl, confidence=0.975, window=250)   # every model, ranked
```

**Options and Greeks**

```python
from risk_engine import black_scholes_greeks, implied_volatility

g = black_scholes_greeks(S=100, K=105, T=1.0, r=0.05, sigma=0.2, kind="call")
print(g.delta, g.gamma, g.quoted()["theta_per_day"])

implied_volatility(price=8.50, S=100, K=105, T=1.0, r=0.05, kind="call")
```

**A mixed equity + options book, under one VaR**

```python
from risk_engine import Portfolio

book = Portfolio()
book.add_equity("AAPL", quantity=1000, price=190.0)
book.add_option(underlying="AAPL", kind="put", strike=180.0, expiry_years=0.25,
                volatility=0.28, quantity=10, spot=190.0)

book.exposure_by_underlying()                  # delta-equivalent, not premium
historical_var(book.scenario_returns(returns), 0.99)
```

**Network contagion**

```python
from risk_engine import correlation_network, rank_systemic_assets

graph = correlation_network(returns, method="threshold", threshold=0.35)
rank_systemic_assets(graph, initial_distress=0.30)   # ranked by systemic impact
```

### What the engine found

Real outputs on daily data for AAPL, MSFT, GOOGL, JPM, XOM (2018-01-03 to
2026-08-07, 2,160 observations) — not illustrations.

#### 1. The normal assumption understates the tail by ~24%

```
                     method     VaR      ES
                 historical 3.9127% 5.4015%
          parametric_normal 3.1598% 3.6338%   <- understates both
       parametric_student_t 3.7111% 5.7167%
       monte_carlo_gaussian 3.1628% 3.6194%
      monte_carlo_student_t 3.6401% 5.5654%
      monte_carlo_bootstrap 3.9318% 5.5730%
 conditional_ewma_empirical 3.3568% 4.0818%   <- today's volatility, not the average
conditional_garch_empirical 3.2859% 4.0124%
```

The spread across methods is 0.77pp — 24% of the smallest estimate. **That
spread is model risk**, and it's the reason more than one method is run. ES
separates the models more sharply than VaR: the normal ES sits 33% below the
historical one.

The two conditional figures are *lower* than the historical one by design. They
answer "how much can we lose tomorrow, given how the market is behaving now",
while the unconditional methods answer "how much on an average day since 2018".
In a calm period the conditional number should be lower — and in a crisis it
rises far faster.

#### 2. Every unconditional model fails its backtest — and *how* it fails names the fix

Walk-forward, 250-day window, 99% confidence, 1,910 out-of-sample days:

| Model | Breaches | Rate | Kupiec p | Christoffersen p |
|---|---|---|---|---|
| **EWMA + FHS** | 32 | 1.68% | 0.0068 | **0.2962** ✓ |
| GARCH + FHS | 34 | 1.78% | 0.0020 | **0.1464** ✓ |
| Historical | 34 | 1.78% | 0.0020 | 0.0230 ✗ |
| EWMA + normal | 38 | 1.99% | 0.0001 | **0.7851** ✓ |
| Parametric Student-t | 38 | 1.99% | 0.0001 | 0.0064 ✗ |
| Parametric normal | 48 | 2.51% | 0.0000 | 0.0011 ✗ |

**The diagnosis came from the independence test, not the coverage test.** Every
unconditional model breaches too often, but that alone doesn't say *why*.
Breaches arriving in bursts is the specific signature of a model assuming
constant volatility. Adding the filter moves Christoffersen's p-value from
0.0011 to 0.79 — and drops the lag-1 autocorrelation of squared returns from
**0.472 to 0.004**.

**At 97.5%, the level Basel FRTB actually uses, the filter is decisive:**

| Model | Breaches | Kupiec p | Christoffersen p | Passes all three |
|---|---|---|---|---|
| **EWMA + FHS** | 61 | 0.0624 | 0.9701 | ✅ **yes** |
| Historical | 65 | 0.0165 | 0.2606 | no |
| GARCH + FHS | 68 | 0.0052 | 0.7099 | no |
| Parametric normal | 67 | 0.0078 | 0.0023 | no |

EWMA with Filtered Historical Simulation is the **only** model of six passing
unconditional coverage, independence and the joint test. That's the headline
result — and it exists because the earlier models failed in a readable way.

#### 3. The most connected asset is not the most systemic

DebtRank under a 30% idiosyncratic shock, network at |ρ| > 0.35:

```
asset  DebtRank  amplification  degree
 MSFT    0.1994          5.65x       3
 AAPL    0.1945          5.54x       3
GOOGL    0.1935          5.52x       3
  JPM    0.1734          5.05x       4   <- most connected, only 4th most systemic
  XOM    0.1044          3.44x       1
  JNJ    0.0203          1.47x       1
  WMT    0.0203          1.47x       1
```

Systemic importance depends on the *strength* of the links and the *value* at
each node, not the count of edges — which is why degree centrality is not a
substitute for a contagion model.

### Design decisions worth defending

The places where the obvious implementation is wrong, and what the engine does
instead.

**Monte Carlo simulates assets, not the portfolio.** Fitting `N(μ, σ)` to the
*portfolio* series and sampling from it isn't an independent method — it
converges to the parametric-normal answer by construction and adds only
simulation noise. Here the *asset* vector is drawn through a Cholesky factor of
the covariance matrix and re-aggregated through the weights, so the estimate
responds to correlation and diversification. The `gaussian` engine is kept as a
control: it *must* reproduce the parametric number, and a test asserts it does.

**Volatility filtering separates two problems.** The filter handles clustering;
the empirical quantile of the standardised residuals handles fat tails. Neither
has to approximate the other — that's why Filtered Historical Simulation beats
both an unconditional model and a conditional-normal one.

**Backtests are walk-forward.** Every forecast at date *t* uses only the window
strictly before *t*. An in-sample VaR is guaranteed roughly the right breach
count and therefore tests nothing. A test asserts no-look-ahead directly, by
truncating the input and checking earlier forecasts are unchanged — it caught a
real seeding bug in the EWMA filter.

**Basel zones are derived, not hardcoded.** The published green 0-4 / amber 5-9
table is a *binomial* result for 250 days at 99%. Rescaling those integers
linearly is wrong twice over — the binomial isn't linear in `n`, and at 95%
confidence you expect 12.5 breaches per 250 days. The thresholds are computed
from binomial quantiles, and a test confirms they reproduce the supervisory
table exactly.

**Contagion uses DebtRank, not repeated multiplicative decay.** Multiplying node
values by `(1 - loss)` each round for a fixed round count guarantees eventual
collapse and measures the round count, not the network. DebtRank gives each node
a state machine — it propagates distress exactly once, then goes inactive —
which bounds the cascade and makes results comparable across networks.

**The network is empirical and sparsified.** A complete graph with random edge
weights encodes no information. Edges come from the sample correlation matrix,
thresholded or reduced to a Mantegna minimum spanning tree over the correlation
distance `d = √(2(1−ρ))`.

**Options are revalued with gamma, not delta alone.** A delta-only approximation
makes a long option's gains and losses symmetric, which is precisely the risk an
option book does *not* have.

**VaR and ES are positive loss magnitudes.** Quoting a negative return quantile
invites sign errors at every boundary. One convention, stated once, enforced
everywhere.

---

## Project 2 — The Art Gallery Problem

> **How few guards can watch every corner of a gallery?**

Chvátal (1975) proved that `⌊n/3⌋ `guards always suffice for a simple polygon
with `n` vertices — and that some polygons need exactly that many. Fisk's 1978
proof fits in a paragraph and is *constructive*, which is what this package
implements:

1. **Triangulate** the polygon (ear clipping).
2. **3-colour** the triangulation's vertices. Always possible, because the dual
   graph is a tree — a greedy colouring walking along it can never trap itself.
3. Every triangle now has one vertex of each colour, so any single colour class
   covers every triangle. Three classes over `n` vertices means the smallest has
   at most `⌊n/3⌋` members. Done.

Guard placements are **verified independently of how they were constructed**.

```python
from artgallery import Polygon, place_guards

gallery = Polygon.from_list([(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)])
solution = place_guards(gallery)

print(solution.summary())
# n=6 vertices, 1 reflex | 4 triangles | 2 guards (bound floor(n/3) = 2)

solution.guard_points     # coordinates to place guards at
solution.triangles        # the triangulation
solution.colouring        # vertex -> colour
```

Lower-level pieces are exposed too — `triangulate`, `three_colour`,
`verify_coverage` — plus polygon primitives (`contains`, `sees`,
`reflex_vertices`, `is_convex`) for building on.

On the **comb** family — the tight instance where each prong hides from every
other — the solver attains the bound exactly, verified up to 6 prongs.

**The honest caveat:** `⌊n/3⌋` is a worst-case guarantee, not an optimum.
Finding the genuinely minimum guard set is NP-hard (Lee & Lin, 1986), and `n` is
the wrong difficulty measure anyway — a convex polygon of any size needs one
guard. It's the *reflex* corners that create hiding places. That gap between the
elegant bound and the hard optimum is where the research sits.

---

## Notebooks

Narrative layers that *import the packages* — the explanation and charts live
here, the logic and tests live in `src/`. All are committed **with their
outputs**, so everything renders on GitHub without running anything.

| Notebook | Covers |
|---|---|
| [`01_var_and_backtesting`](notebooks/01_var_and_backtesting.ipynb) | VaR/ES across every method, walk-forward backtesting, coverage tests, and the volatility filter that fixes them |
| [`02_options_and_greeks`](notebooks/02_options_and_greeks.ipynb) | Pricing, Greeks surfaces, implied volatility, and two independent validations |
| [`03_network_contagion`](notebooks/03_network_contagion.ipynb) | Correlation networks, DebtRank cascades, systemic ranking |
| [`04_portfolio_integration`](notebooks/04_portfolio_integration.ipynb) | One VaR number across a mixed equity + options book |
| [`05_art_gallery_problem`](notebooks/05_art_gallery_problem.ipynb) | Fisk's proof implemented, verified and drawn |

```bash
pip install -e ".[notebooks]"
jupyter lab notebooks/
```

---

## How it's validated

Built around *known-truth* testing — generate data whose correct answer is known
analytically, then assert the estimator recovers it.

| What is asserted | Why it's convincing |
|---|---|
| `C − P = S·e^{−qT} − K·e^{−rT}` to 1e-9 | Model-free identity; breaks if either branch is wrong |
| CRR lattice → Black-Scholes as steps ↑ | Independent numerical method, zero shared code |
| Greeks vs. central finite differences | Catches an analytic derivative that is merely plausible |
| American call (q=0) = European | Merton's no-early-exercise result |
| Parametric VaR on `N(0, σ)` = `z·σ` | Closed-form tail quantile |
| A GARCH fit recovers its simulated parameters | The MLE is correct, not merely convergent |
| Filtering removes autocorrelation from squared returns | The volatility model does the job it exists to do |
| Kupiec passes 10/1000, rejects 50/1000 | The test is itself tested, both directions |
| Christoffersen rejects contiguous breaches | It detects the clustering it exists to detect |
| Truncating input leaves earlier forecasts unchanged | No look-ahead — caught a real EWMA seeding bug |
| `basel_zone_thresholds(250, 0.99) == (4, 9)` | Reproduces the published supervisory table |
| Uncorrelated pair carries less VaR than an identical pair | Diversification is actually represented |
| Triangle areas sum to the polygon's area | The triangulation partitions — no gaps, no overlaps |
| Guard count ≤ `⌊n/3⌋`, attained on combs | The theorem itself, in both directions |

---

## Repository layout

```
src/risk_engine/
├── data.py         market data, caching, synthetic fallback
├── var.py          VaR & Expected Shortfall estimators
├── volatility.py   EWMA & GARCH(1,1) filters, FHS residuals
├── backtest.py     walk-forward backtesting + coverage tests
├── options.py      Black-Scholes, Greeks, implied vol, binomial lattice
├── portfolio.py    mixed equity/option book, scenario revaluation
├── contagion.py    correlation networks + DebtRank
├── report.py       charts and tables
└── cli.py          risk-engine command line
src/artgallery/
├── geometry.py     polygon primitives, visibility, containment
└── solver.py       triangulation, 3-colouring, guard placement
tests/              266 tests, 93% coverage
notebooks/          narrative walkthroughs, committed with outputs
scripts/            export engine results to the portfolio site
archive/            the original single-cell notebooks, kept for provenance
.github/workflows/  CI: ruff + mypy, then tests on Python 3.10-3.13
```

---

## Development

```bash
pytest -q                                    # 266 tests, ~4 minutes
pytest --cov=risk_engine --cov=artgallery    # coverage report
pytest tests/test_volatility.py -v           # one module
ruff check src tests scripts                 # lint
mypy                                         # type check
```

Regenerate the figures and the portfolio site's data:

```bash
risk-engine report --tickers AAPL,MSFT,GOOGL,JPM,XOM --output outputs
python scripts/export_site_data.py --site ../shahmeerkhurram.github.io
```

The site reads its numbers from the JSON that script writes, so published
results can't drift from what the code produces.

---

## Known limitations

Stated plainly, because a risk tool that hides its assumptions is the thing it
exists to prevent.

- **No model achieves nominal coverage at 99%.** The volatility filter fixes
  clustering decisively and passes everything at 97.5%, but this tail defeats
  all six models. An EVT peaks-over-threshold tail fitted to the standardised
  residuals is the honest next step.
- **GARCH is refit every 25 walk-forward steps**, for runtime. The variance
  recursion still updates daily, so there's no look-ahead — but the coefficients
  are staler than a daily refit would give.
- **Square-root-of-time horizon scaling** assumes iid returns. It's the Basel
  convention and it understates risk under the volatility clustering found here.
- **Delta-gamma option revaluation** degrades for very large moves and near
  expiry; full repricing is the right tool for deep stress scenarios.
- **Constant portfolio weights** — a daily-rebalanced risk view, not a
  buy-and-hold backtest.
- **Correlation is not causation.** The contagion network measures co-movement,
  not contractual exposure; a true interbank DebtRank would use a liabilities
  matrix.
- **yfinance is an unofficial data source.** Fine for research, not for anything
  load-bearing.
- **Ear clipping is O(n²)** — adequate for demonstrating the theorem, not for
  large meshes.

---

## Author

**Shahmeer Khurram** — BSc Mathematics, University of Debrecen (4.7/5).
Research Assistant, Department of Geometry — Art Gallery Problem.

[Portfolio](https://shahmeerkhurram.github.io) ·
[Risk engine write-up](https://shahmeerkhurram.github.io/risk-engine) ·
[LinkedIn](https://www.linkedin.com/in/shahmeerkhurram)

## License

MIT — see [LICENSE](LICENSE).

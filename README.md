# Distributionally Robust End-to-End Portfolio Construction

Replication and extension of Costa & Iyengar (2022/2023), implementing a distributionally robust (DR) end-to-end learning system for portfolio construction. The system jointly trains a return prediction network with a differentiable optimization layer, where the DR decision layer solves a minimax problem reformulated as a tractable convex minimization via convex duality — enabling gradient-based training through the optimization layer using `cvxpylayers`.

The full project report is available in [`Term Paper.pdf`](./Term_Paper.pdf).

---

## Papers

**Primary reference (replicated):**
> Costa, G. and Iyengar, G.N. (2023). *Distributionally Robust End-to-End Portfolio Construction.* Quantitative Finance, 23(10):1465–1482. [[arXiv]](https://arxiv.org/abs/2206.05134)

**Secondary reference:**
> Uysal, A.S., Li, X., and Mulvey, J.M. (2021). *End-to-End Risk Budgeting Portfolio Optimization with Neural Networks.* arXiv:2107.04636. [[arXiv]](https://arxiv.org/abs/2107.04636)

---

## Our Contributions

Beyond replicating the original 5 experiments from Costa & Iyengar (2023), we extended the framework with alternative optimization objectives in the decision layer:

**Markowitz-family objectives** (`e2edro/e2edro_Risk.py`):
- Mean-variance (DPP-compliant via Cholesky factorization: `z⊤Σz = ‖L⊤z‖²`)
- Max Sharpe ratio (via efficient frontier search over a grid of risk-aversion parameters)

**CVaR-based objectives** (`e2edro/e2edro_Risk.py`):
- Minimum CVaR (Rockafellar-Uryasev formulation — naturally DPP-compliant)
- Mean-CVaR (return vs tail-risk trade-off)

**Risk-only objectives** (`e2edro/e2edro_Markov.py`):
- Minimum variance with learned covariance matrix
- Risk budgeting (convex approximation from Bruder & Roncalli 2012)
- Risk parity (iterative equal-risk-contribution via the risk budgeting layer)

All objectives are formulated to satisfy Disciplined Parametrized Programming (DPP) rules, enabling a single KKT factorization to be cached and reused during backpropagation.

---

## System Architecture

```
Input features (x)
        ↓
Prediction layer (θ) — fully connected NN, single hidden layer, leaky ReLU
        ↓
  Predicted returns (ŷ) and prediction errors (ε)
        ↓
DR Decision layer — solves:
  min  f_δ(z, c, λ, ξ) − γ · ŷᵀz     [convex duality reformulation of minimax]
  z∈Z, λ≥0, ξ, c
        ↓
  Optimal portfolio z*
        ↓
Task loss: Sharpe ratio over next v=13 periods
        ↓
Backprop → update θ, γ, δ jointly
```

The risk appetite γ and robustness parameter δ are both **learned end-to-end** from data, eliminating manual calibration. The ambiguity set uses the Hellinger distance φ-divergence by default (also supports total variation distance).

---

## Experiments

### Experiments 1–5: Replication (historical S&P 500 data)

**Data:** weekly returns of 20 S&P 500 stocks (Jan 2000 – Oct 2021) from AlphaVantage, with 8 Fama-French factors as predictive features. Train/test split: 60/40. Rolling retraining every ~2 years (4 windows). Performance window: 13 weeks. Prediction errors window: T=104 weeks.

| Exp | Description | Key result |
|---|---|---|
| 1 | General evaluation: EW vs PO vs Base vs Nominal vs DR | DR achieves Sharpe 1.30 vs 1.05 (EW), 0.88 (PO) |
| 2 | Isolated effect of learning δ | Robustness improves performance even without learning δ |
| 3 | Isolated effect of learning γ | Learning γ alone is insufficient; robustness is essential |
| 4 | Isolated effect of learning θ | Learning θ is only beneficial when combined with a risk measure |
| 5 | NN prediction layers on synthetic data (10 assets, 5 features, 1200 obs) | DR consistently outperforms Nominal across linear, 2-layer, and 3-layer networks |

### Extension experiments: alternative optimization objectives (ETF universe)

**Data:** 7 ETFs (VTI, IWM, AGG, LQD, MUB, DBC, GLD), daily returns 2010–2021. Features: 5-day lagged returns, rolling mean and variance over 10/20/30 days. Train/test split: 60/40. Portfolio rebalanced every 14 trading days.

| Model | Return | Volatility | Sharpe |
|---|---|---|---|
| Equal Weight | 0.41% | 0.56% | 0.74 |
| E2E Min-Variance | 0.41% | 0.60% | 0.68 |
| Min-Variance (historical) | 0.21% | 0.56% | 0.38 |
| Risk Budget | 0.24% | 1.02% | 0.24 |
| E2E Mean-Variance | 0.42% | 1.16% | 0.36 |
| **E2E Max Sharpe** | **0.28%** | **0.35%** | **0.78** |

Key finding: direct Sharpe ratio optimization yields the best risk-adjusted returns; model performance is more constrained by risk estimation quality than return prediction accuracy.

---

## Repository Structure

```
dr-e2e-portfolio-optimization/
│
├── main.py                      # Experiments 1–5 (replication)
├── E2E_main_jupyter.ipynb       # Interactive notebook: replication + extension experiments
├── Term_Paper.pdf               # Final project report
├── portfolio_comparison.png     # Out-of-sample wealth evolution (Experiment 1)
│
└── e2edro/                      # Core library
    ├── __init__.py
    ├── e2edro.py                # e2e_net class: training loop, CV, rolling backtest
    ├── DataLoad.py              # Data loading: AlphaVantage, Fama-French, synthetic
    ├── BaseModels.py            # Baselines: equal weight, predict-then-optimize, gamma_range
    ├── RiskFunctions.py         # Deviation risk measures: p_var (variance), p_mad (MAD)
    ├── LossFunctions.py         # Task loss functions: sharpe_loss, single_period_loss
    ├── PortfolioClasses.py      # SlidingWindow dataset, backtest, CrossVal objects
    ├── PlotFunctions.py         # Wealth evolution, Sharpe bar charts, fin_table
    ├── e2edro_Markov.py         # Our extension: risk-only optimization layers
    │                            #   (min-variance, risk budgeting, risk parity)
    └── e2edro_Risk.py           # Our extension: return-based optimization layers
                                 #   (mean-variance, max Sharpe, min-CVaR, mean-CVaR)
```

---

## Setup

### Requirements

Python 3.8 or later is recommended.

Key packages: `torch`, `cvxpy`, `cvxpylayers`, `pandas`, `numpy`, `matplotlib`, `scipy`

### AlphaVantage API Key

Experiments 1–4 use historical stock data from AlphaVantage. A free academic API key is available at [www.alphavantage.co](https://www.alphavantage.co).

Set it in `main.py`:
```python
AV_key = "YOUR_API_KEY"
```

If `use_cache = True` (default), pre-computed results are loaded from `cache/` and no API key is required.

### Fama-French factors

Downloaded automatically from Kenneth French's data library via `e2edro/DataLoad.py`.

---

## Running

```bash
# Experiments 1–5: load cached results and produce tables/plots
python main.py

# Interactive exploration of all experiments including extensions
jupyter notebook E2E_main_jupyter.ipynb
```

To retrain from scratch, set `use_cache = False` in `main.py`. Full training takes several hours on CPU due to rolling 4-window cross-validation.

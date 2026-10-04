# Rough-volatility deep hedging

This project develops and evaluates deep hedging policies under a calibrated rough-Bergomi market model. It contains executable result notebooks, reusable simulation and hedging code, compact cloud-run provenance, and the CSV tables supporting every displayed result. Raw market downloads, model checkpoints, and workflow artifacts are intentionally excluded.

## Start here

| Path | Purpose |
| --- | --- |
| `notebooks/01_mathematical_foundations.ipynb` | Rough-volatility, pricing, and CVaR foundations. |
| `notebooks/02_synthetic_playground.ipynb` | Completed L4 synthetic recipe study; one figure per cell. |
| `notebooks/03_real_data.ipynb` | Simulated-training evidence followed by historical S&P 500 replay. |
| `notebooks/README.md` | Theory, reading order, and evidence boundary. |
| `rough_hedge/` | Reusable simulation, risk, hedging, and training implementation. |
| `scripts/t4_recipes.py` | Recipe-study trainer (small profile). |
| `scripts/t4_recipes_large.py` | Recipe-study trainer as run for the large profile. |
| `scripts/exp_nb03_hedging.py` | Immutable training and replay pipeline. |
| `results/` | Compact L4 training and replay provenance. |

## Results at a glance

In the synthetic L4 study, 15 recipe × architecture cells are evaluated per profile under common accounting. The large profile has lower mean CVaR95 than the small profile for all 15 paired cells; the best recorded large-profile unit has CVaR95 0.05742 versus 0.06195 for Black--Scholes delta on the common simulated evaluation. Notebook 03 then separates this simulated evidence from the historical S&P 500 replay, including daily and intraday rehedging trade-offs.

These are conditional experimental and historical-replay results, not investment advice or a claim of future trading performance.

## Reproduction

Use Python 3.12 with NumPy, SciPy, pandas, matplotlib, and PyTorch. Obtain market inputs independently with:

```bash
python scripts/fetch_data.py
```

Run the immutable replay pipeline with:

```bash
python scripts/exp_nb03_hedging.py train --run-id RUN_ID --workers 4
python scripts/exp_nb03_hedging.py replay --run-id RUN_ID
python scripts/exp_nb03_hedging.py figures --run-id RUN_ID
```

The compact evidence already used by the notebooks is under `results/nb02/t4_stats.json` and `results/nb03/runs/l4-20261003-full-04/`.

## References

1. Bayer, C., Friz, P., and Gatheral, J. *Pricing under rough volatility*. Quantitative Finance, 2016.
2. Buehler, H., Gonon, L., Teichmann, J., and Wood, B. *Deep Hedging*. Quantitative Finance, 2019.
3. Rockafellar, R. T., and Uryasev, S. *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2000.

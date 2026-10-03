# Rough-volatility deep hedging

This project develops and evaluates deep hedging policies under a calibrated rough-Bergomi market model.  The release includes three executable notebooks, production implementations, compact provenance, and real-window replay artifacts. Raw market inputs and model checkpoints are not redistributed.

## Index

| Path | Contents |
| --- | --- |
| `notebooks/01_mathematical_foundations.ipynb` | Rough-volatility, pricing, and CVaR foundations. |
| `notebooks/02_synthetic_playground.ipynb` | Simulation-based hedging experiments. |
| `notebooks/03_real_data.ipynb` | Market-data calibration and real-window evaluation. |
| `notebooks/README.md` | Notebook map and the detailed training diagnostics gallery. |
| `rough_hedge/` | Reusable simulation, risk, hedging, and training code. |
| `scripts/exp_nb03_hedging.py` | Immutable training, replay, and visualization CLI. |
| `scripts/t4_recipes.py` | Recipe-study trainer with per-epoch diagnostics. |
| `scripts/gen_figures_t4.py` | Recipe-study renderer for Figures 5.1--5.16. |
| `results/nb02/t4_stats.json` | Compact L4 recipe-study findings and statistics. |
| `results/nb03/runs/l4-20261003-full-04/` | Compact L4 training and replay provenance. |

## Reproduction

Use Python 3.12 with NumPy, SciPy, pandas, matplotlib, and PyTorch. Obtain market inputs independently with `python scripts/fetch_data.py`; the included manifest records their expected hashes. Run the immutable replay pipeline with:

```bash
python scripts/exp_nb03_hedging.py train --run-id RUN_ID --workers 4
python scripts/exp_nb03_hedging.py replay --run-id RUN_ID
python scripts/exp_nb03_hedging.py figures --run-id RUN_ID
```

## L4 recipe-study visualizations

The study compares five training recipes, three learned hedge architectures, three seeds, and two training profiles. Every panel below has its compact CSV source beside the PNG.

### Mathematical foundations (Notebook 01)

<table><tr>
<td width="25%"><img src="figures/fig-1.0.png" alt="Simulated variance mean"><br><sub><b>Figure 1.0.</b> Simulated variance mean with ±4 SE band; <code>fig-1.0.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-1.2.png" alt="Roughness recovery"><br><sub><b>Figure 1.2.</b> Roughness recovery by series type; <code>fig-1.2.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-1.6.png" alt="Rough Bergomi paths"><br><sub><b>Figure 1.6.</b> Rough-Bergomi variance and spot paths; <code>fig-1.6.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-1.7.png" alt="Volterra power law"><br><sub><b>Figure 1.7.</b> Volterra variance power-law check; <code>fig-1.7.csv</code>.</sub></td>
</tr></table>

### Immediate synthetic and real-data results (Notebooks 02--03)

<table><tr>
<td width="25%"><img src="figures/fig-5.1.png" alt="CVaR leaderboard"><br><sub><b>Figure 5.1.</b> CVaR95 leaderboard; <code>fig-5.1.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-5.2.png" alt="paired CVaR"><br><sub><b>Figure 5.2.</b> Paired CVaR gaps to delta; <code>fig-5.2.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-5.3.png" alt="learning curves"><br><sub><b>Figure 5.3.</b> Validation learning curves; <code>fig-5.3.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-5.4.png" alt="compute frontier"><br><sub><b>Figure 5.4.</b> Compute--risk frontier; <code>fig-5.4.csv</code>.</sub></td>
</tr></table>

<table><tr>
<td width="25%"><img src="figures/fig-5.13.png" alt="gradient updates"><br><sub><b>Figure 5.13.</b> Gradient/update diagnostics; <code>fig-5.13.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-5.14.png" alt="classical benchmark"><br><sub><b>Figure 5.14.</b> Classical benchmark comparison; <code>fig-5.14.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-5.15.png" alt="loss tail"><br><sub><b>Figure 5.15.</b> Full loss distribution and tail; <code>fig-5.15.csv</code>.</sub></td>
<td width="25%"><img src="figures/fig-5.16.png" alt="tail risk"><br><sub><b>Figure 5.16.</b> Risk across tail levels; <code>fig-5.16.csv</code>.</sub></td>
</tr></table>

## Real-market replay visualizations

<table><tr>
<td width="50%"><img src="figures/fig-4.5-l4-20261003-full-04-analysis.png" alt="Daily S&P 500 and VIX replay"><br><sub><b>Figure 4.5.</b> Daily real-window losses, regime CVaR95, and paired differences. Data: <code>fig-4.5-l4-20261003-full-04-analysis.csv</code>.</sub></td>
<td width="50%"><img src="figures/fig-4.6-l4-20261003-full-04-analysis.png" alt="Intraday rehedging replay"><br><sub><b>Figure 4.6.</b> Intraday CVaR-frequency trade-off, turnover/cost, and failure paths. Data: <code>fig-4.6-l4-20261003-full-04-analysis.csv</code>.</sub></td>
</tr></table>

The daily replay spans 191 non-overlapping S&P 500/VIX windows; the intraday replay spans 57 complete five-day hourly windows. Learned policies are trained on simulated paths only. Both recorded prefix-information leakage probes pass exactly.

## References

1. Bayer, C., Friz, P., and Gatheral, J. *Pricing under rough volatility*. Quantitative Finance, 2016.
2. Buehler, H., Gonon, L., Teichmann, J., and Wood, B. *Deep Hedging*. Quantitative Finance, 2019.
3. Rockafellar, R. T., and Uryasev, S. *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2000.

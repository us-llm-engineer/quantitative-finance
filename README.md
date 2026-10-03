# Rough-volatility deep hedging

This release trains 35 deep-hedging policies on calibrated simulated rough-Bergomi paths and evaluates them only on held-out real S&P 500/VIX windows.  The official L4 run is `l4-20261003-full-04`; no raw market data or model checkpoints are included.

Run the reproducible pipeline with:

```bash
python scripts/exp_nb03_hedging.py train --run-id RUN_ID --workers 4
python scripts/exp_nb03_hedging.py replay --run-id RUN_ID
python scripts/exp_nb03_hedging.py figures --run-id RUN_ID
```

The compact release artifacts are the training provenance, daily and intraday replay summaries, and the new run-scoped figures. Figure 4.5 compares loss distributions and CVaR95 with 95% block-bootstrap intervals across low/high VIX regimes. Figure 4.6 shows the CVaR/turnover trade-off across intraday rehedging frequencies and the three worst real windows. Colors are identified in each panel legend.

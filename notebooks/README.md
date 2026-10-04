# Notebook guide: theory and evidence

## Reading order

1. `01_mathematical_foundations.ipynb` establishes the rough-volatility model, the Volterra-kernel viewpoint, pricing quantities, and tail-risk convention.
2. `02_synthetic_playground.ipynb` is the completed L4 simulation study. It presents one result per cell so the inferential comparison, optimisation behavior, distributional evidence, and resource use can be reviewed independently.
3. `03_real_data.ipynb` connects simulated training to a held-out historical S&P 500 replay under the same hedging-accounting convention.

## Theory in brief

Under rough Bergomi, the forward variance is driven by a Volterra process with Hurst parameter \(H < 1/2\). Smaller \(H\) produces rougher, less regular volatility paths. A hedge is assessed through its self-financing terminal hedging error, including proportional trading cost.

For a loss variable \(L\), the study uses conditional value at risk at level \(\alpha\):

\[
\operatorname{CVaR}_{\alpha}(L) = \mathbb{E}[L \mid L \geq \operatorname{VaR}_{\alpha}(L)].
\]

Lower CVaR means a smaller average loss in the worst \(1-\alpha\) tail. The Rockafellar--Uryasev representation used during training is

\[
\min_v\left(v + \frac{1}{1-\alpha}\,\mathbb{E}[(L-v)_+]\right).
\]

The simulation notebook uses common test paths for paired comparisons against Black--Scholes delta. The historical notebook retains that idea where possible, but its overlapping realised windows require block-bootstrap uncertainty summaries and do not justify independent-window inference.

## Evidence boundary

Figures and their compact CSV source tables live in the executable notebooks. The repository deliberately excludes model checkpoints, raw market downloads, and workflow artifacts. Recreate source data with `scripts/fetch_data.py`, run the replay pipeline with `scripts/exp_nb03_hedging.py`, and use the versioned compact JSON files under `results/` to audit reported values.

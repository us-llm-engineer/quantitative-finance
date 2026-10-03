#!/usr/bin/env python3
"""Figures 5.1-5.16 (L4 recipe study) and results/nb02/t4_stats.json.

Usage: python3 scripts/gen_figures_t4.py --root results/nb02/t4_l4 [--out-root <repo root>]
       [--boot-cache <json>]   optional cache of the 90 paired bootstraps (development only)

--root holds results_{large,small}/ (or {large,small}/). Every number is computed from those files.
Paired tests: rough_hedge.stats.paired_bootstrap(learned, classical, n_boot=2000, seed=0), so a
negative estimate means learned CVaR95 is BELOW the classical hedge. Bootstraps run in a
ProcessPoolExecutor (max 4 workers).
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys
import re
import csv
import json
import base64
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import scipy.stats as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rough_hedge.stats import paired_bootstrap, holm

warnings.filterwarnings('ignore', category=RuntimeWarning)

RECIPE_ORDER = ['R-GPU', 'R-NTBN', 'R-SIG', 'R-DH', 'R-HOR']
ARM_ORDER = ['FFN', 'GRU', 'sig2']
ARM_COLORS = {'FFN': '#1f77b4', 'GRU': '#ff7f0e', 'sig2': '#2ca02c'}
CLASSICAL_ORDER = ['bs_delta', 'leland', 'ww_band']
CLASSICAL_COLORS = {'bs_delta': '#303030', 'leland': '#808080', 'ww_band': '#b0b0b0'}
RECIPE_MARKERS = {'R-GPU': 'o', 'R-NTBN': 's', 'R-SIG': '^', 'R-DH': 'D', 'R-HOR': 'v'}
RECIPE_LS = {'R-GPU': '-', 'R-NTBN': '--', 'R-SIG': '-.', 'R-DH': ':', 'R-HOR': (0, (5, 1, 1, 1, 1, 1))}
N_BOOT = 2000
SEED = 0
MAX_WORKERS = 2  # paired_bootstrap peaks at ~1.8 GB RSS (2000 x 20000 index and sort arrays); 3-4 workers swap on this 7 GB machine
PNL_Q = ['0.01', '0.05', '0.25', '0.5', '0.75', '0.95', '0.99']


# ----------------------------------------------------------------------------- loading
def load_units(root, profile):
    files = sorted(root.glob(f'results_{profile}/R-*.json')) or sorted(root.glob(f'{profile}/R-*.json'))
    units = {}
    for f in files:
        if re.fullmatch(r'R-[A-Za-z]+_[A-Za-z0-9]+_\d+', f.stem):
            with open(f) as fp:
                units[f.stem] = json.load(fp)
    if not units:
        raise FileNotFoundError(f'no unit JSONs for profile {profile} under {root}')
    return units


def load_classical(root, profile):
    for p in (root / f'results_{profile}' / 'classical.json', root / profile / 'classical.json'):
        if p.exists():
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(f'classical.json for {profile}')


def parse_key(key):
    recipe, arm, seed = key.rsplit('_', 2)
    return recipe, arm, int(seed)


def dec_f32(d):
    return np.frombuffer(base64.b64decode(d['b64']), np.float32)


def dec_f16(s):
    return np.frombuffer(base64.b64decode(s), np.float16).astype(np.float64)


def cells(units):
    """(recipe, arm) -> sorted list of (seed, unit)."""
    out = defaultdict(list)
    for k, u in units.items():
        r, a, s = parse_key(k)
        out[(r, a)].append((s, u))
    for v in out.values():
        v.sort(key=lambda t: t[0])
    return out


def mean_ci(vals):
    v = np.asarray(vals, float)
    m = v.mean()
    if len(v) < 2:
        return m, 0.0
    return m, float(sps.sem(v) * sps.t.ppf(0.975, len(v) - 1))


def ep_series(unit, field):
    return np.array([np.nan if e.get(field) is None else e[field]
                     for e in unit['histories']['monitor']['epochs']], float)


def stack(arrs):
    n = max(len(a) for a in arrs)
    return np.array([np.concatenate([a, np.full(n - len(a), np.nan)]) for a in arrs], float)


def best_cell(units):
    """(recipe, arm) with the lowest seed-mean test CVaR95."""
    best, bv = None, np.inf
    for (r, a), lst in cells(units).items():
        m = np.mean([u['stats']['cvar_95'] for _, u in lst])
        if m < bv:
            best, bv = (r, a), m
    return best, bv


# ----------------------------------------------------------------------------- output
class Out:
    def __init__(self, out_root):
        self.fig_dir = out_root / 'figures'
        self.fig_dir.mkdir(exist_ok=True)

    def save(self, fig, n, rows, cols):
        fig.savefig(self.fig_dir / f'fig-5.{n}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        with open(self.fig_dir / f'fig-5.{n}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        return rows


def arm_handles():
    return [Line2D([], [], color=ARM_COLORS[a], lw=2.5, label=a) for a in ARM_ORDER]


def recipe_ls_handles():
    return [Line2D([], [], color='k', ls=RECIPE_LS[r], lw=1.8, label=r.replace('R-', '')) for r in RECIPE_ORDER]


def recipe_marker_handles(filled=True):
    return [Line2D([], [], color='k', marker=RECIPE_MARKERS[r], ls='', mfc='k' if filled else 'none',
                   label=r.replace('R-', '')) for r in RECIPE_ORDER]


# ----------------------------------------------------------------------------- bootstraps
def _boot_worker(args):
    tag, learned, classical = args
    r = paired_bootstrap(learned, classical, n_boot=N_BOOT, seed=SEED)
    return tag, dict(estimate=float(r.estimate), lo=float(r.lo), hi=float(r.hi), pvalue=float(r.pvalue), se=float(r.se))


def compute_boots(all_units, all_classical, cache_path=None):
    """Return {'profile|unitkey': result}. learned first, classical second => negative = learned better."""
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)
    jobs = []
    for profile, units in all_units.items():
        bs = dec_f32(all_classical[profile]['classical_arms']['bs_delta']['per_path_losses'])
        for k, u in sorted(units.items()):
            jobs.append((f'{profile}|{k}', dec_f32(u['per_path_losses']), bs))
    res = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for tag, r in ex.map(_boot_worker, jobs):
            res[tag] = r
            print(f'  bootstrap {len(res)}/{len(jobs)} {tag}', flush=True)
    if cache_path:
        with open(cache_path, 'w') as f:
            json.dump(res, f)
    return res


def cell_table(profile, units, boots):
    """Per recipe x arm: mean of per-seed estimates, max per-seed p, Holm over the cells of the profile."""
    rows = []
    for r in RECIPE_ORDER:
        for a in ARM_ORDER:
            lst = cells(units).get((r, a), [])
            if not lst:
                continue
            res = [boots[f'{profile}|{r}_{a}_{s}'] for s, _ in lst]
            rows.append(dict(profile=profile, recipe=r, arm=a, n_seeds=len(lst),
                             delta_cvar95=float(np.mean([x['estimate'] for x in res])),
                             p_raw=float(max(x['pvalue'] for x in res)),
                             ci_lo_mean=float(np.mean([x['lo'] for x in res])),
                             ci_hi_mean=float(np.mean([x['hi'] for x in res]))))
    adj, rej = holm([x['p_raw'] for x in rows])
    for x, p, rj in zip(rows, adj, rej):
        x['p_holm'] = float(p)
        x['holm_reject'] = bool(rj)
    return rows


# ----------------------------------------------------------------------------- figures
def fig1(o, U, C):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for ax, prof in zip(axes, ['small', 'large']):
        y = 0
        ticks, labels = [], []
        cl = cells(U[prof])
        for r in RECIPE_ORDER:
            for a in ARM_ORDER:
                v = [u['stats']['cvar_95'] for _, u in cl[(r, a)]]
                m, ci = mean_ci(v)
                ax.errorbar(m, y, xerr=ci, fmt=RECIPE_MARKERS[r], color=ARM_COLORS[a], ms=6, capsize=3)
                ticks.append(y)
                labels.append(f'{r[2:]}/{a}')
                rows.append(dict(profile=prof, recipe=r, arm=a, n_seeds=len(v), mean=m, ci_lo=m - ci, ci_hi=m + ci))
                y += 1
            y += 0.5
        for j, c in enumerate(CLASSICAL_ORDER):
            v = C[prof]['classical_arms'][c]['stats']['cvar_95']
            ax.axvline(v, color=CLASSICAL_COLORS[c], ls='--', lw=1.5, label=f'{c} ({v:.4f})')
            rows.append(dict(profile=prof, recipe='classical', arm=c, mean=v, ci_lo=v, ci_hi=v))
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('test CVaR95 (mean over 3 seeds, 95% t-interval)')
        ax.set_title(f'{prof} profile')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return o.save(fig, 1, rows, ['profile', 'recipe', 'arm', 'n_seeds', 'mean', 'ci_lo', 'ci_hi'])


def fig2(o, U, tables):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    vmax = max(abs(x['delta_cvar95']) for t in tables.values() for x in t)
    for ax, prof in zip(axes, ['small', 'large']):
        M = np.full((3, 5), np.nan)
        for x in tables[prof]:
            M[ARM_ORDER.index(x['arm']), RECIPE_ORDER.index(x['recipe'])] = x['delta_cvar95']
            rows.append(x)
        im = ax.imshow(M, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(5))
        ax.set_xticklabels([r[2:] for r in RECIPE_ORDER])
        ax.set_yticks(range(3))
        ax.set_yticklabels(ARM_ORDER)
        for x in tables[prof]:
            i, j = ARM_ORDER.index(x['arm']), RECIPE_ORDER.index(x['recipe'])
            col = 'white' if abs(x['delta_cvar95']) > 0.6 * vmax else 'black'
            ax.text(j, i, f"{x['delta_cvar95']:+.4f}\np={x['p_holm']:.3f}", ha='center', va='center', fontsize=8, color=col)
        ax.set_title(f'{prof}: $\\Delta$CVaR95 learned $-$ bs_delta\n(negative = learned better; cell text: estimate, Holm-adjusted p over 15 cells)', fontsize=10)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return o.save(fig, 2, rows, ['profile', 'recipe', 'arm', 'n_seeds', 'delta_cvar95', 'ci_lo_mean', 'ci_hi_mean',
                                 'p_raw', 'p_holm', 'holm_reject'])


def fig3(o, U):
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, r in zip(axes.flat, RECIPE_ORDER):
        for a in ARM_ORDER:
            lst = cells(U['large'])[(r, a)]
            V = stack([np.array(u['histories']['val_cvar'], float) for _, u in lst])
            S = stack([np.cumsum([e['steps'] for e in u['histories']['monitor']['epochs']]).astype(float) for _, u in lst])
            step, med, lo, hi = np.nanmedian(S, 0), np.nanmedian(V, 0), np.nanmin(V, 0), np.nanmax(V, 0)
            ax.plot(step, med, color=ARM_COLORS[a], lw=1.8)
            ax.fill_between(step, lo, hi, color=ARM_COLORS[a], alpha=0.2)
            for i in range(len(step)):
                rows.append(dict(profile='large', recipe=r, arm=a, epoch=i, step=step[i], val_cvar95_median=med[i],
                                 val_cvar95_min=lo[i], val_cvar95_max=hi[i]))
        ax.set_title(r[2:])
        ax.set_xlabel('optimizer step')
        ax.set_ylabel('val CVaR95')
        ax.grid(alpha=0.3)
    ax = axes.flat[5]
    ax.axis('off')
    ax.legend(handles=arm_handles(), loc='center', fontsize=12, title='median over 3 seeds, band = min-max')
    fig.tight_layout()
    return o.save(fig, 3, rows, ['profile', 'recipe', 'arm', 'epoch', 'step', 'val_cvar95_median', 'val_cvar95_min', 'val_cvar95_max'])


def fig4(o, U):
    rows = []
    fig, ax = plt.subplots(figsize=(9, 6))
    for prof in ['large', 'small']:
        for k, u in sorted(U[prof].items()):
            r, a, s = parse_key(k)
            gs = float(sum(u['histories']['wall_s']))
            cv = u['stats']['cvar_95']
            ax.scatter(gs, cv, marker=RECIPE_MARKERS[r], s=60, edgecolor=ARM_COLORS[a],
                       facecolor=ARM_COLORS[a] if prof == 'large' else 'none', linewidths=1.5, alpha=0.85)
            rows.append(dict(profile=prof, recipe=r, arm=a, seed=s, cvar95=cv, gpu_seconds=gs))
    ax.set_xscale('log')
    ax.set_xlabel('GPU-seconds per unit (sum of epoch wall_s, log)')
    ax.set_ylabel('final test CVaR95')
    ax.grid(alpha=0.3, which='both')
    h = arm_handles() + recipe_marker_handles() + [
        Line2D([], [], marker='o', color='gray', mfc='gray', ls='', label='large (filled)'),
        Line2D([], [], marker='o', color='gray', mfc='none', ls='', label='small (hollow)')]
    ax.legend(handles=h, fontsize=8, ncol=2, loc='best')
    fig.tight_layout()
    return o.save(fig, 4, rows, ['profile', 'recipe', 'arm', 'seed', 'cvar95', 'gpu_seconds'])


def fig5(o, U):
    rows = []
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, r in zip(axes, RECIPE_ORDER):
        bsz = None
        any_line = False
        for a in ARM_ORDER:
            lst = cells(U['large'])[(r, a)]
            bsz = lst[0][1]['config']['batch_size']
            S = stack([ep_series(u, 'grad_noise_scale_B_simple') for _, u in lst])
            med = np.nanmedian(S, 0)
            ok = ~np.isnan(med)
            if ok.any():
                any_line = True
                ax.semilogy(np.arange(len(med))[ok], med[ok], color=ARM_COLORS[a], lw=1.8)
            for i in np.where(ok)[0]:
                rows.append(dict(profile='large', recipe=r, arm=a, epoch=int(i), b_simple_median=med[i], batch_size=bsz))
        if any_line:
            ax.axhline(bsz, color='k', ls='--', lw=1)
            ax.annotate(f'batch = {bsz}', xy=(0.02, bsz), xycoords=('axes fraction', 'data'), xytext=(0, -11), textcoords='offset points', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'B_simple undefined\n(1 step per epoch)', transform=ax.transAxes, ha='center', va='center')
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(r[2:])
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('B_simple (log)')
    fig.legend(handles=arm_handles() + [Line2D([], [], color='k', ls='--', label='batch size')],
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return o.save(fig, 5, rows, ['profile', 'recipe', 'arm', 'epoch', 'b_simple_median', 'batch_size'])


def fig6(o, U):
    rows = []
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for ax, r in zip(axes, RECIPE_ORDER):
        for a in ARM_ORDER:
            lst = cells(U['large'])[(r, a)]
            v = np.nanmedian(stack([ep_series(u, 'ru_threshold_v') for _, u in lst]), 0)
            q = np.nanmedian(stack([ep_series(u, 'probe_var95') for _, u in lst]), 0)
            ax.plot(v, color=ARM_COLORS[a], ls='-', lw=1.8)
            ax.plot(q, color=ARM_COLORS[a], ls='--', lw=1.8)
            for i in range(len(v)):
                rows.append(dict(profile='large', recipe=r, arm=a, epoch=i, ru_threshold_v=v[i], probe_var95=q[i]))
        ax.set_title(r[2:])
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('v / VaR95 (median over seeds)')
    fig.legend(handles=arm_handles() + [Line2D([], [], color='k', ls='-', label='v (RU threshold)'),
                                        Line2D([], [], color='k', ls='--', label='probe VaR95')],
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return o.save(fig, 6, rows, ['profile', 'recipe', 'arm', 'epoch', 'ru_threshold_v', 'probe_var95'])


def fig7(o, U):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (field, ttl) in zip(axes, [('delta_dist_mean', 'all t'), ('delta_dist_t0', 't = 0')]):
        for r in RECIPE_ORDER:
            for a in ARM_ORDER:
                lst = cells(U['large'])[(r, a)]
                m = np.nanmedian(stack([ep_series(u, field) for _, u in lst]), 0)
                ax.plot(m, color=ARM_COLORS[a], ls=RECIPE_LS[r], lw=1.5)
                for i in range(len(m)):
                    rows.append(dict(profile='large', recipe=r, arm=a, epoch=i, which=field, value=m[i]))
        ax.set_title(f'mean |$\\delta_\\theta - \\delta_{{BS}}$|, {ttl}')
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('mean absolute distance (median over seeds)')
    fig.legend(handles=arm_handles() + recipe_ls_handles(), loc='lower center', ncol=8, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    return o.save(fig, 7, rows, ['profile', 'recipe', 'arm', 'epoch', 'which', 'value'])


def fig8(o, U):
    rows = []
    (br, ba), _ = best_cell(U['large'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, a in zip(axes, ARM_ORDER):
        u = cells(U['large'])[(br, a)][0][1]
        snaps = u['histories']['monitor']['snapshots']
        eps = sorted(int(k) for k in snaps)
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(eps)))
        for c, e in zip(cmap, eps):
            L = np.sort(dec_f16(snaps[str(e)]))
            n = len(L)
            sf = (n - np.arange(n)) / n  # P(L >= x_i)
            ax.semilogy(L, sf, color=c, lw=1.5, label=f'epoch {e}')
            for x, s in zip(L[::8], sf[::8]):
                rows.append(dict(profile='large', recipe=br, arm=a, seed=0, snapshot_epoch=e, loss=x, survival=s))
        ax.set_title(f'{a}')
        ax.set_xlabel('probe loss x')
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('P(L > x)')
    fig.suptitle(f'{br[2:]} (best recipe by large-profile cell-mean CVaR95), seed 0, 4,096 probe paths', y=1.02)
    fig.tight_layout()
    return o.save(fig, 8, rows, ['profile', 'recipe', 'arm', 'seed', 'snapshot_epoch', 'loss', 'survival'])


def fig9(o, U):
    rows = []
    (br, ba), _ = best_cell(U['large'])
    bands = [(0, 6, '1-99%', 0.25), (1, 5, '5-95%', 0.45), (2, 4, '25-75%', 0.70)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, a in zip(axes, ARM_ORDER):
        u = cells(U['large'])[(br, a)][0][1]
        eps = u['histories']['monitor']['epochs']
        X = np.arange(len(eps))
        Q = {q: np.array([e['pnl_quantiles'][q] for e in eps], float) for q in PNL_Q}
        for lo, hi, lab, al in bands:
            ax.fill_between(X, Q[PNL_Q[lo]], Q[PNL_Q[hi]], color=ARM_COLORS[a], alpha=al, lw=0, label=lab)
        ax.plot(X, Q['0.5'], color='k', lw=1.5, label='median')
        for i in X:
            for q in PNL_Q:
                rows.append(dict(profile='large', recipe=br, arm=a, seed=0, epoch=int(i), quantile=q, pnl=Q[q][i]))
        ax.set_title(a)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('P&L quantile')
    fig.suptitle(f'{br[2:]} (best recipe), seed 0', y=1.02)
    fig.tight_layout()
    return o.save(fig, 9, rows, ['profile', 'recipe', 'arm', 'seed', 'epoch', 'quantile', 'pnl'])


def fig10(o, U):
    rows = []
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = 0
    ticks, labels = [], []
    for r in RECIPE_ORDER:
        for a in ARM_ORDER:
            ms = {}
            for prof in ['small', 'large']:
                v = [u['stats']['cvar_95'] for _, u in cells(U[prof])[(r, a)]]
                ms[prof] = mean_ci(v)
                rows.append(dict(profile=prof, recipe=r, arm=a, n_seeds=len(v), cvar95_mean=ms[prof][0], cvar95_ci=ms[prof][1]))
            ax.plot([x - 0.2, x + 0.2], [ms['small'][0], ms['large'][0]], color=ARM_COLORS[a], lw=1.5)
            ax.errorbar(x - 0.2, ms['small'][0], yerr=ms['small'][1], fmt='o', mfc='none', color=ARM_COLORS[a], capsize=3)
            ax.errorbar(x + 0.2, ms['large'][0], yerr=ms['large'][1], fmt='o', color=ARM_COLORS[a], capsize=3)
            ticks.append(x)
            labels.append(f'{r[2:]}/{a}')
            x += 1
        x += 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel('test CVaR95 (mean over seeds, 95% t-interval)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(handles=arm_handles() + [Line2D([], [], marker='o', color='gray', mfc='none', ls='', label='small (left)'),
                                      Line2D([], [], marker='o', color='gray', ls='', label='large (right)')],
              fontsize=8, ncol=5, loc='upper right')
    fig.tight_layout()
    return o.save(fig, 10, rows, ['profile', 'recipe', 'arm', 'n_seeds', 'cvar95_mean', 'cvar95_ci'])


def anova_components(units):
    R, A = len(RECIPE_ORDER), len(ARM_ORDER)
    cl = cells(units)
    n = min(len(cl[(r, a)]) for r in RECIPE_ORDER for a in ARM_ORDER)
    Y = np.array([[[u['stats']['cvar_95'] for _, u in cl[(r, a)]][:n] for a in ARM_ORDER] for r in RECIPE_ORDER])  # R,A,n
    g = Y.mean()
    mr, ma, mc = Y.mean((1, 2)), Y.mean((0, 2)), Y.mean(2)
    ss_r = A * n * np.sum((mr - g) ** 2)
    ss_a = R * n * np.sum((ma - g) ** 2)
    ss_ra = n * np.sum((mc - mr[:, None] - ma[None, :] + g) ** 2)
    ss_e = np.sum((Y - mc[:, :, None]) ** 2)
    tot = np.sum((Y - g) ** 2)
    assert abs(ss_r + ss_a + ss_ra + ss_e - tot) < 1e-9 * max(tot, 1e-12), 'ANOVA sums of squares do not add up'
    return {'recipe': ss_r / tot, 'arm': ss_a / tot, 'recipe x arm': ss_ra / tot, 'seed residual': ss_e / tot}, n


def fig11(o, U):
    rows = []
    comps = ['recipe', 'arm', 'recipe x arm', 'seed residual']
    cols = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, prof in enumerate(['small', 'large']):
        fr, n = anova_components(U[prof])
        bottom = 0
        for c, col in zip(comps, cols):
            v = 100 * fr[c]
            ax.bar(i, v, bottom=bottom, color=col, label=c if i == 0 else None)
            if v > 3:
                ax.text(i, bottom + v / 2, f'{v:.1f}%', ha='center', va='center', fontsize=9, color='white')
            bottom += v
            rows.append(dict(profile=prof, component=c, variance_percent=v, n_seeds=n))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['small', 'large'])
    ax.set_ylabel('% of total variance of CVaR95 (two-way ANOVA SS)')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    return o.save(fig, 11, rows, ['profile', 'component', 'variance_percent', 'n_seeds'])


def parse_ts(s):
    return datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()


def num(s):
    return float(re.sub(r'[^0-9.eE+\-]', '', s.strip().split()[0]))


def fig12(o, root):
    gpu = root / 'results_large' / 'gpu_all.csv'
    log = root / 'results_large' / 'batch.log'
    if not gpu.exists():
        gpu, log = root / 'large' / 'gpu_all.csv', root / 'large' / 'batch.log'
    # gpu_all.csv holds two interleaved sampler streams (a ~30 kB run of NUL bytes sits between them), so
    # drop unparsable rows and sort by time.
    recs = []
    n_bad = 0
    with open(gpu, errors='ignore') as f:
        next(f)
        for line in f:
            row = line.replace('\x00', '').strip().split(',')
            try:
                recs.append((parse_ts(row[0]), num(row[1]), num(row[3]) / 1024.0, num(row[4])))
            except (ValueError, IndexError):
                n_bad += 1
    recs.sort()
    t, util, mem, pw = (list(x) for x in zip(*recs))
    print(f'  gpu csv: {len(recs)} rows parsed, {n_bad} unparsable lines dropped', flush=True)
    t = np.array(t)
    t0 = t[0]
    minutes = (t - t0) / 60.0
    # batch.log: only the last batch_all run covers gpu_all.csv
    lines = open(log).read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith('batch_all start')]
    seg = lines[starts[-1]:]
    # the log header timestamps are UTC and nvidia-smi is local time: align the first launch with the first GPU row
    ev = []
    for l in seg:
        m = re.match(r'\s+launch (\S+) .* t\+(\d+)s', l)
        if m:
            ev.append((int(m.group(2)), +1, m.group(1)))
        m = re.match(r'\s+(\S+): rc=(-?\d+) t\+(\d+)s', l)
        if m:
            ev.append((int(m.group(3)), -1, m.group(1)))
    ev.sort(key=lambda e: (e[0], e[1]))
    first_launch = min(e[0] for e in ev if e[1] == 1)
    # approximate alignment: first launch <-> first GPU row (+ the gap between batch start and first row is not logged)
    ev_min = np.array([(e[0] - first_launch) / 60.0 for e in ev])
    ev_d = np.array([e[1] for e in ev])
    flight = np.array([ev_d[ev_min <= m].sum() for m in minutes])
    rows = [dict(timestamp=datetime.fromtimestamp(t[i]).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3], minutes=minutes[i],
                 gpu_util_pct=util[i], memory_gib=mem[i], power_w=pw[i], units_in_flight=int(flight[i])) for i in range(len(t))]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(minutes, util, color='#1f77b4', lw=1.2)
    ax.set_ylabel('GPU utilisation (%)', color='#1f77b4')
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.fill_between(minutes, flight, step='post', color='gray', alpha=0.25)
    ax2.set_ylabel('units in flight (shaded)')
    ax2.set_ylim(0, 20)
    axes[1].plot(minutes, mem, color='#2ca02c', lw=1.2)
    axes[1].set_ylabel('GPU memory (GiB)')
    axes[2].plot(minutes, pw, color='#d62728', lw=1.2)
    axes[2].set_ylabel('power (W)')
    axes[2].set_xlabel('minutes since first nvidia-smi row')
    for a in axes:
        a.grid(alpha=0.3)
    fig.suptitle('L4 timeline, large profile (units in flight from batch.log launch/rc lines; alignment to the GPU clock is approximate)',
                 fontsize=10, y=0.99)
    fig.tight_layout()
    return o.save(fig, 12, rows, ['timestamp', 'minutes', 'gpu_util_pct', 'memory_gib', 'power_w', 'units_in_flight'])


def fig13(o, U):
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, a in zip(axes, ARM_ORDER):
        key = f'R-HOR_{a}_0'
        if key not in U['large']:
            raise KeyError(f'{key} missing from large profile')
        eps = U['large'][key]['histories']['monitor']['epochs']
        mods = list(eps[0]['layer_grad_share'].keys())
        M = np.array([[e['layer_grad_share'][m] for e in eps] for m in mods], float)
        im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
        ax.set_yticks(range(len(mods)))
        ax.set_yticklabels(mods)
        ax.set_xlabel('epoch')
        ax.set_title(f'{a} (R-HOR, seed 0)')
        fig.colorbar(im, ax=ax, label='gradient share')
        for i, m in enumerate(mods):
            for j in range(len(eps)):
                rows.append(dict(profile='large', recipe='R-HOR', arm=a, seed=0, epoch=j, module=m, grad_share=M[i, j]))
    fig.tight_layout()
    return o.save(fig, 13, rows, ['profile', 'recipe', 'arm', 'seed', 'epoch', 'module', 'grad_share'])


def fig14(o, U, C):
    rows = []
    fig, ax = plt.subplots(figsize=(9, 6))
    for prof in ['large', 'small']:
        for k, u in sorted(U[prof].items()):
            r, a, s = parse_key(k)
            to, cv = u['stats']['turnover'], u['stats']['cvar_95']
            ax.scatter(to, cv, marker=RECIPE_MARKERS[r], s=50, edgecolor=ARM_COLORS[a], linewidths=1.5,
                       facecolor=ARM_COLORS[a] if prof == 'large' else 'none', alpha=0.85)
            rows.append(dict(profile=prof, kind='learned', recipe=r, arm=a, seed=s, turnover=to, cvar95=cv))
        for c in CLASSICAL_ORDER:
            st = C[prof]['classical_arms'][c]['stats']
            ax.scatter(st['turnover'], st['cvar_95'], marker='*', s=220, edgecolor='k', linewidths=1,
                       facecolor=CLASSICAL_COLORS[c] if prof == 'large' else 'none')
            if prof == 'large':
                ax.annotate(c, (st['turnover'], st['cvar_95']), fontsize=8, xytext=(-8, 9), textcoords='offset points', ha='right')
            rows.append(dict(profile=prof, kind='classical', recipe='classical', arm=c, seed='', turnover=st['turnover'], cvar95=st['cvar_95']))
    ax.set_xlabel('turnover')
    ax.set_ylabel('test CVaR95')
    ax.grid(alpha=0.3)
    h = arm_handles() + recipe_marker_handles() + [
        Line2D([], [], marker='o', color='gray', mfc='gray', ls='', label='large (filled)'),
        Line2D([], [], marker='o', color='gray', mfc='none', ls='', label='small (hollow)'),
        Line2D([], [], marker='*', color='k', ls='', ms=12, label='classical (grey fill)')]
    ax.legend(handles=h, fontsize=8, ncol=2, loc='best')
    fig.tight_layout()
    return o.save(fig, 14, rows, ['profile', 'kind', 'recipe', 'arm', 'seed', 'turnover', 'cvar95'])


def best_unit(units):
    k = min(units, key=lambda k: units[k]['stats']['cvar_95'])
    return k, units[k]


def fig15(o, U, C):
    rows = []
    bk, bu = best_unit(U['large'])
    curves = [(f'best learned: {bk}', 'learned', dec_f32(bu['per_path_losses']), 'C3', 2.0)]
    for c in CLASSICAL_ORDER:
        curves.append((c, c, dec_f32(C['large']['classical_arms'][c]['per_path_losses']), CLASSICAL_COLORS[c], 1.6))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax_i, ax in enumerate(axes):
        for lab, tag, L, col, lw in curves:
            L = np.sort(L)
            n = len(L)
            sf = (n - np.arange(n)) / n
            ax.semilogy(L, sf, color=col, lw=lw, label=lab)
            if ax_i == 0:
                for x, s in zip(L[::4], sf[::4]):
                    rows.append(dict(curve=tag, unit=bk if tag == 'learned' else tag, loss=x, survival=s))
        ax.set_xlabel('loss x')
        ax.set_ylabel('P(L > x)')
        ax.grid(alpha=0.3, which='both')
    ref = np.quantile(curves[1][2], 0.9)
    axes[1].set_xlim(ref, max(c[2].max() for c in curves))
    axes[1].set_ylim(1e-4, 0.12)
    axes[0].set_title('all 20,000 test paths')
    axes[1].set_title('tail zoom (loss above the bs_delta 90% quantile)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return o.save(fig, 15, rows, ['curve', 'unit', 'loss', 'survival'])


def fig16(o, U):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (field, ttl) in zip(axes, [('grad_norm_mean', 'mean gradient norm'), ('update_to_weight_ratio', 'update-to-weight ratio')]):
        for r in RECIPE_ORDER:
            for a in ARM_ORDER:
                lst = cells(U['large'])[(r, a)]
                m = np.nanmedian(stack([ep_series(u, field) for _, u in lst]), 0)
                ax.semilogy(m, color=ARM_COLORS[a], ls=RECIPE_LS[r], lw=1.5)
                for i in range(len(m)):
                    rows.append(dict(profile='large', recipe=r, arm=a, epoch=i, metric=field, value=m[i]))
        ax.set_title(ttl)
        ax.set_xlabel('epoch')
        ax.set_ylabel(f'{ttl} (log, median over seeds)')
        ax.grid(alpha=0.3, which='both')
    fig.legend(handles=arm_handles() + recipe_ls_handles(), loc='lower center', ncol=8, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    return o.save(fig, 16, rows, ['profile', 'recipe', 'arm', 'epoch', 'metric', 'value'])


# ----------------------------------------------------------------------------- findings
def generate_findings(U, C, boots, tables):
    F = []
    # 1. learned below bs_delta (Holm over the 15 cells of each profile)
    num_ = {}
    for prof in ['large', 'small']:
        t = tables[prof]
        sig = [x for x in t if x['holm_reject'] and x['delta_cvar95'] < 0]
        b = min(t, key=lambda x: x['delta_cvar95'])
        num_[prof] = dict(n_cells=len(t), n_cells_learned_below_after_holm=len(sig),
                          best_cell=f"{b['recipe']}_{b['arm']}", best_cell_delta_cvar95=b['delta_cvar95'],
                          best_cell_p_raw=b['p_raw'], best_cell_p_holm=b['p_holm'],
                          bs_delta_cvar95=C[prof]['classical_arms']['bs_delta']['stats']['cvar_95'])
    bk, bu = best_unit(U['large'])
    r = boots[f'large|{bk}']
    num_['best_unit_large'] = dict(unit=bk, learned_cvar95=bu['stats']['cvar_95'], delta_cvar95_vs_bs_delta=r['estimate'],
                                   ci=[r['lo'], r['hi']], pvalue=r['pvalue'])
    F.append(dict(falsifier='learned_below_bs_delta',
                  description='At least one recipe x arm cell (large profile) is below bs_delta in CVaR95 after Holm; '
                              'estimates are learned minus bs_delta, negative = learned better',
                  is_true=bool(num_['large']['n_cells_learned_below_after_holm'] > 0), numbers=num_))
    # 2. large better than small (paired over the 15 recipe x arm cell means)
    cl, cs = [], []
    for r_ in RECIPE_ORDER:
        for a in ARM_ORDER:
            cl.append(np.mean([u['stats']['cvar_95'] for _, u in cells(U['large'])[(r_, a)]]))
            cs.append(np.mean([u['stats']['cvar_95'] for _, u in cells(U['small'])[(r_, a)]]))
    cl, cs = np.array(cl), np.array(cs)
    large_mean, small_mean = float(cl.mean()), float(cs.mean())
    diff = small_mean - large_mean
    tt = sps.ttest_rel(cs, cl)
    F.append(dict(falsifier='large_better_than_small',
                  description='Large profile has lower CVaR95 than small (paired over 15 cells, t-test)',
                  is_true=bool(diff > 0 and tt.pvalue < 0.05),
                  numbers=dict(large_mean_cvar95=large_mean, small_mean_cvar95=small_mean, small_minus_large=diff,
                               relative_change_pct_of_small=100.0 * diff / small_mean,
                               n_cells_large_lower=int((cl < cs).sum()), n_cells=len(cl),
                               paired_t=float(tt.statistic), paired_p=float(tt.pvalue))))
    # 3. B_simple vs batch size
    ratios = {}
    for r_ in RECIPE_ORDER:
        vals, bsz = [], None
        for a in ARM_ORDER:
            for _, u in cells(U['large'])[(r_, a)]:
                bsz = u['config']['batch_size']
                v = ep_series(u, 'grad_noise_scale_B_simple')
                vals.extend(v[~np.isnan(v)])
        ratios[r_] = dict(batch_size=bsz, median_B_simple=float(np.median(vals)) if vals else None,
                          ratio_B_simple_over_batch=float(np.median(vals) / bsz) if vals else None)
    rv = [x['ratio_B_simple_over_batch'] for x in ratios.values() if x['ratio_B_simple_over_batch'] is not None]
    F.append(dict(falsifier='batch_size_within_10x_of_B_simple',
                  description='median B_simple within [0.1, 10] x batch size for every recipe with a defined B_simple '
                              '(below 0.1: batch wasteful; above 10: larger batch would help; R-NTBN has 1 step/epoch, undefined)',
                  is_true=bool(all(0.1 <= x <= 10 for x in rv)), numbers=ratios))
    # 4. v tracks VaR95
    cors, gaps = [], []
    for u in U['large'].values():
        v, q = ep_series(u, 'ru_threshold_v'), ep_series(u, 'probe_var95')
        ok = ~np.isnan(v) & ~np.isnan(q)
        if ok.sum() > 2:
            cors.append(float(np.corrcoef(v[ok], q[ok])[0, 1]))
        gaps.append(abs(v[ok][-1] - q[ok][-1]) / abs(q[ok][-1]))
    F.append(dict(falsifier='v_tracks_var95',
                  description='RU threshold v within 10% of probe VaR95 at the last epoch (median over 45 large units)',
                  is_true=bool(np.median(gaps) <= 0.10),
                  numbers=dict(median_final_relative_gap=float(np.median(gaps)), max_final_relative_gap=float(np.max(gaps)),
                               mean_epoch_correlation=float(np.nanmean(cors)), n_units=len(gaps))))
    return F


def main():
    ap = argparse.ArgumentParser(description='Generate figures 5.1-5.16')
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out-root', type=Path, default=project_root)
    ap.add_argument('--boot-cache', type=Path, default=None)
    args = ap.parse_args()
    root, out_root = args.root, args.out_root
    if not root.exists():
        raise ValueError(f'root not found: {root}')
    U = {p: load_units(root, p) for p in ['large', 'small']}
    C = {p: load_classical(root, p) for p in ['large', 'small']}
    print(f"loaded {len(U['large'])} large and {len(U['small'])} small units", flush=True)
    boots = compute_boots(U, C, args.boot_cache)
    print(f'{len(boots)} paired bootstraps done (n_boot={N_BOOT}, seed={SEED})', flush=True)
    tables = {p: cell_table(p, U[p], boots) for p in ['large', 'small']}
    o = Out(out_root)
    figs = {
        '5.1': fig1(o, U, C), '5.2': fig2(o, U, tables), '5.3': fig3(o, U), '5.4': fig4(o, U), '5.5': fig5(o, U),
        '5.6': fig6(o, U), '5.7': fig7(o, U), '5.8': fig8(o, U), '5.9': fig9(o, U), '5.10': fig10(o, U),
        '5.11': fig11(o, U), '5.12': fig12(o, root), '5.13': fig13(o, U), '5.14': fig14(o, U, C),
        '5.15': fig15(o, U, C), '5.16': fig16(o, U),
    }
    findings = generate_findings(U, C, boots, tables)
    summary = {k: (v if len(v) <= 400 else dict(n_rows=len(v), csv=f'figures/fig-{k}.csv')) for k, v in figs.items()}
    out = dict(n_boot=N_BOOT, seed=SEED, paired_bootstrap_sign='learned minus classical (negative = learned better)',
               holm_family='15 recipe x arm cells per profile', figures=summary, findings=findings)
    (out_root / 'results' / 'nb02').mkdir(parents=True, exist_ok=True)
    with open(out_root / 'results' / 'nb02' / 't4_stats.json', 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print('wrote', out_root / 'results' / 'nb02' / 't4_stats.json')


if __name__ == '__main__':
    main()

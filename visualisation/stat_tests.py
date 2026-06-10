"""All significance tests for the main figure (Panels B/C, 128-image not_all8 set).

Two test families, each at its appropriate unit:
  * 6 metrics (comp_corr, rsa, voxel_within, voxel_cross, exact, balanced):
    per-subject paired-t, n=4 — diffs + Cohen's d_z + uncorrected p, with
    Benjamini-Hochberg FDR across the six metrics (within each pairwise contrast).
  * silhouette: combined (pooled, UMAP-10d) silhouette, tested by a paired image
    bootstrap (B resamples over the 128 images) — delta, 95% CI, two-sided p.

Reads the eval outputs from <results>/ (alignment_eval_*.pkl + silh_latents_*.pkl,
produced by evaluation/evaluate_methods.py) and writes data/figure_stats.json
(silhouette bars + all significance brackets) consumed by render_figure.py.
Needs umap/sklearn for the silhouette UMAP-10d reduction.

  python stat_tests.py        # honours $CCN_RESULTS_DIR, else evaluation/results
"""
import sys, os, pickle, contextlib, json
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.join(os.path.dirname(HERE), 'evaluation')
RESULTS = os.environ.get('CCN_RESULTS_DIR', os.path.join(_EVAL, 'results'))   # metrics pkls + silh_latents
SUBJ = [1, 2, 5, 7]
NB = 5000
PK = {m: pickle.load(open(f'{RESULTS}/alignment_eval_{m}_32d_not_all8_streams.pkl', 'rb'))
      for m in ['vae', 'srm', 'procrustes']}


# ----------------------------- n=4 paired-t helpers -----------------------------
def collapse_to_subjects(M):
    M = np.asarray(M, float); n = M.shape[0]
    return np.array([np.mean([M[s, j] for j in range(n) if j != s]) for s in range(n)])


def paired_compare(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); diffs = a - b; n = len(diffs)
    t_p = float(stats.ttest_rel(a, b).pvalue)
    sd = diffs.std(ddof=1); d_z = float(diffs.mean() / sd) if sd > 0 else float('nan')
    return {'n': n, 't_p': t_p, 'd_z': d_z, 'diffs': diffs}


def bh_fdr(pvals):
    p = np.asarray(pvals, float); n = p.size; order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1); return out


def cc_matrix(d):   # component-correlation matrix (renamed key, back-compat)
    return np.asarray(d.get('component_corr_matrix', d.get('prediction_matrix')), float)


def arrays(m):
    d = PK[m]
    return {
        'comp_corr':    collapse_to_subjects(cc_matrix(d)),
        'rsa':          collapse_to_subjects(np.asarray(d['rsa_matrix_pearson'], float)),
        'voxel_within': np.array([d['recon_per_subject'][s]['voxel_correlation'] for s in SUBJ]),
        'voxel_cross':  np.array([d['cross_trial']['cross_trial_per_subject'][s]['mean_voxel_corr'] for s in SUBJ]),
        'exact_match':  np.asarray(d['decoding']['per_subject_exact'], float),
        'balanced_acc': np.asarray(d['decoding']['per_subject_balanced'], float),
    }


METRICS6 = ['comp_corr', 'rsa', 'voxel_within', 'voxel_cross', 'exact_match', 'balanced_acc']
A = {m: arrays(m) for m in ['vae', 'srm', 'procrustes']}


# ----------------------------- silhouette bootstrap -----------------------------
def silhouette_bootstrap():
    sys.path.insert(0, _EVAL)
    import metrics as _M
    from metrics import _reduce_if_needed, calculate_multilabel_silhouette_hybrid as silh
    # The bars are UMAP-10d silhouettes; the bootstrap MUST use the same reduction.
    # _reduce_if_needed silently falls back to PCA if UMAP can't import (e.g. numba/numpy
    # conflict on some nodes) -> refuse to run rather than report PCA-space stats.
    assert _M.UMAP is not None, ('UMAP unavailable on this node/env -> _reduce_if_needed would '
                                 'silently use PCA. Run on a node with working UMAP (e.g. GPU partition).')
    emb, lab, nimg, nsub = {}, {}, None, None
    point = {}
    ref_labels = None
    for m in ['vae', 'srm', 'procrustes']:
        d = pickle.load(open(f'{RESULTS}/silh_latents_{m}_32d_not_all8_streams.pkl', 'rb'))
        lats = [np.asarray(x, float) for x in d['latents']]
        labels = np.asarray(d['labels'], float)
        # all three methods must share the SAME 128 images (same order) for paired bootstrap
        if ref_labels is None:
            ref_labels = labels
        else:
            assert np.array_equal(labels, ref_labels), f'{m} labels differ -> image pairing invalid'
        nsub = len(lats); nimg = lats[0].shape[0]
        X = np.vstack(lats); L = np.vstack([labels] * nsub)
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            Xr = _reduce_if_needed(X); point[m] = float(silh(Xr, L))
        # UMAP availability is asserted above, so `point` is a genuine UMAP-10d silhouette
        # (never a silent PCA fallback). It may differ from the stored pkl bar if that bar was
        # computed with a different UMAP *version* -> warn, and treat the live value as canonical
        # so the bar and bracket come from one consistent UMAP.
        bar = float(PK[m]['silhouette']['silhouette_combined'])
        if abs(point[m] - bar) >= 0.01:
            print(f'  [warn] {m}: live UMAP silhouette {point[m]:.4f} != stored pkl bar {bar:.4f} '
                  f'(stored bar is an older UMAP version; using the live consistent value)')
        emb[m], lab[m] = Xr, L
    rows = lambda imgs: np.concatenate([imgs + s * nimg for s in range(nsub)])
    rng = np.random.default_rng(42)
    boot = {m: np.empty(NB) for m in emb}
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        for b in range(NB):
            r = rows(rng.integers(0, nimg, size=nimg))
            for m in emb:
                boot[m][b] = silh(emb[m][r], lab[m][r])
    return point, boot, nimg, nsub


# --------------------------------- report ---------------------------------------
PAIRS = [('vae', 'srm'), ('vae', 'procrustes'), ('srm', 'procrustes')]
METHOD_IDX = {'vae': 0, 'srm': 1, 'procrustes': 2}
IDX_PAIRS = [(METHOD_IDX[a], METHOD_IDX[b]) for a, b in PAIRS]


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'


print('#' * 78)
print('# MAIN FIGURE  —  ALL Panel B/C significance tests  (32d, not_all8, 128 images, subj 1/2/5/7)')
print('#' * 78)

metric_stars = {name: {} for name in METRICS6}   # name -> {(i,j): star}  from the n=4 paired-t
for ma, mb in PAIRS:
    print(f'\n{"="*78}\n {ma.upper()} vs {mb.upper()}   (diff = {ma} - {mb})\n{"="*78}')
    print('  6 metrics — per-subject paired-t (n=4).  Lead with diffs + d_z; p uncorrected; q=BH(6).')
    res = {name: paired_compare(A[ma][name], A[mb][name]) for name in METRICS6}
    q = bh_fdr([res[name]['t_p'] for name in METRICS6])
    print(f'    {"metric":13} {"diffs [S1,S2,S5,S7]":28} {"d_z":>7} {"t_p":>8} {"q_BH":>7} {"":>4}')
    for name, qq in zip(METRICS6, q):
        r = res[name]
        metric_stars[name][(METHOD_IDX[ma], METHOD_IDX[mb])] = stars(r['t_p'])
        print(f'    {name:13} {str(np.round(r["diffs"],4).tolist()):28} {r["d_z"]:>7.2f} '
              f'{r["t_p"]:>8.4f} {qq:>7.4f} {stars(r["t_p"]):>4}')

print(f'\n{"="*78}\n SILHOUETTE  —  combined (pooled, UMAP-10d) silhouette, paired image-bootstrap\n{"="*78}')
point, boot, nimg, nsub = silhouette_bootstrap()
ci = lambda x: np.percentile(x, [2.5, 97.5])
print(f'  B={NB} resamples, {nimg} images x {nsub} subjects.  Point estimate = full-sample silhouette.')
print('  per-method combined silhouette + bootstrap 95% CI:')
for m in ['vae', 'srm', 'procrustes']:
    lo, hi = ci(boot[m])
    se = boot[m].std(ddof=1)
    print(f'    {m:11} {point[m]:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   (boot SE {se:.4f})')
print('  pairwise differences (paired bootstrap):')
print(f'    {"pair":16} {"Δ":>8} {"95% CI":>20} {"p_2sided":>9} {"":>4}')
silh_pairs = {}
for ma, mb in PAIRS:
    diff = boot[ma] - boot[mb]; lo, hi = ci(diff)
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    print(f'    {ma+"-"+mb:16} {diff.mean():>8.4f}   [{lo:>7.4f},{hi:>7.4f}]   {p:>9.4f} {stars(p):>4}')
    silh_pairs[f'{ma}-{mb}'] = {'delta': float(diff.mean()), 'ci': [float(lo), float(hi)],
                                'p': float(p), 'stars': stars(p)}

# --- export silhouette bars + ALL significance brackets for render_figure.py ---
# silhouette bar = live UMAP-10d combined silhouette; error bar (in render) = bootstrap 95% CI.
silhouette_bars = {m: {'point': float(point[m]),
                       'ci': [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))]}
                   for m in ['vae', 'srm', 'procrustes']}
# Panel B brackets: 0 silhouette (image-bootstrap), 1 exact-match (n=4 t), 2 balanced (n=4 t)
sig_B = {
    0: [[METHOD_IDX[a], METHOD_IDX[b], silh_pairs[f'{a}-{b}']['stars']] for a, b in PAIRS],
    1: [[i, j, metric_stars['exact_match'][(i, j)]] for (i, j) in IDX_PAIRS],
    2: [[i, j, metric_stars['balanced_acc'][(i, j)]] for (i, j) in IDX_PAIRS],
}
# Panel C brackets: 0 comp_corr, 1 rsa, 2 voxel_within, 3 voxel_cross (all n=4 paired-t)
sig_C = {k: [[i, j, metric_stars[name][(i, j)]] for (i, j) in IDX_PAIRS]
         for k, name in enumerate(['comp_corr', 'rsa', 'voxel_within', 'voxel_cross'])}

_OUT = os.path.join(HERE, 'data', 'figure_stats.json')
os.makedirs(os.path.dirname(_OUT), exist_ok=True)
with open(_OUT, 'w') as fh:
    json.dump({'silhouette_bars': silhouette_bars,
               'sig_B': {str(k): v for k, v in sig_B.items()},
               'sig_C': {str(k): v for k, v in sig_C.items()}}, fh, indent=2)
print(f'  wrote {_OUT}')

print('\nSTATS_DONE')

"""Render Fig. 3 — cross-subject neural prediction (voxel-wise correlation).

For each ordered pair i→j of subjects 1, 2, 5, 7, subject i's fMRI for the 128 test
images is encoded into the shared space and decoded into subject j's vertices (MED-VAE:
j's decoder; SRM/Procrustes: j's inverse transform and PCA). Each bar is the Pearson r
between predicted and measured responses across images, averaged over j's vertices;
grouped bars of the 12 pairs for MED-VAE / SRM / Procrustes.

Input (produced by the release pipeline — see README):
  <results>/alignment_eval_{vae,srm,procrustes}_32d_not_all8_streams.pkl   (evaluation/evaluate_methods.py)
  key `fmri_prediction_matrix` (row = source i, column = target j; the diagonal is not plotted).
<results> defaults to $CCN_RESULTS_DIR, else evaluation/results.

Usage:
  python render_figure3.py [--results <dir>] [--out figure3_crosssubj.eps]   # also writes .pdf/.png
"""
import os
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.environ.get(
    'CCN_RESULTS_DIR', os.path.join(os.path.dirname(HERE), 'evaluation', 'results'))

N_COMP, MODE, DATASET = 32, 'not_all8', 'streams'
SUBJECTS = [1, 2, 5, 7]
METHOD_COLORS = {'MED-VAE': '#8E86C0', 'SRM': '#C99494', 'Procrustes': '#88B8A9'}


def load_pairwise(results_dir, method):
    """{(src, tgt): voxel_corr} from the eval's fmri_prediction_matrix."""
    d = pickle.load(open(os.path.join(results_dir, f'alignment_eval_{method}_{N_COMP}d_{MODE}_{DATASET}.pkl'), 'rb'))
    M = d.get('fmri_prediction_matrix')
    subs = list(d.get('subjects', SUBJECTS))
    if M is None:
        raise SystemExit(f'{method}: no fmri_prediction_matrix (subjects={subs})')
    if not set(SUBJECTS) <= set(subs):   # a missing pair would otherwise be plotted as 0
        raise SystemExit(f'{method}: results for subjects {subs}; Fig. 3 needs subjects {SUBJECTS}')
    M = np.asarray(M, float)
    return {(src, tgt): float(M[i, j]) for i, src in enumerate(subs)
            for j, tgt in enumerate(subs) if src != tgt}


def create_figure(vae, srm, proc, out_path):
    pairs = [(i, j) for i in SUBJECTS for j in SUBJECTS if i != j]
    labels = [f'{i}→{j}' for i, j in pairs]
    vae_v = np.array([vae.get(p, 0.0) for p in pairs])
    srm_v = np.array([srm.get(p, 0.0) for p in pairs])
    proc_v = np.array([proc.get(p, 0.0) for p in pairs])

    print('pairs :', labels)
    print('VAE   :', np.round(vae_v, 4).tolist())
    print('SRM   :', np.round(srm_v, 4).tolist())
    print('Proc  :', np.round(proc_v, 4).tolist())
    print(f'means   VAE {vae_v.mean():.4f}   SRM {srm_v.mean():.4f}   Procrustes {proc_v.mean():.4f}')

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pairs)); w = 0.25
    ax.bar(x - w, vae_v, w, label='MED-VAE', color=METHOD_COLORS['MED-VAE'])
    ax.bar(x,     srm_v, w, label='SRM', color=METHOD_COLORS['SRM'])
    ax.bar(x + w, proc_v, w, label='Procrustes', color=METHOD_COLORS['Procrustes'])

    ax.set_ylabel('Voxel-wise Correlation (r)', fontsize=26, fontweight='bold')
    ax.set_xlabel('Subject Pairs', fontsize=26, fontweight='bold')
    ax.set_title('Cross-subject Neural Prediction', fontsize=30, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=20)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2); ax.spines['bottom'].set_linewidth(1.2)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(fontsize=18, loc='upper right', framealpha=0.9)

    allv = np.concatenate([vae_v, srm_v, proc_v]); pos = allv[allv > 0]
    ax.set_ylim(max(0.0, pos.min() - 0.05), pos.max() + 0.05)

    plt.tight_layout()
    for ext in ('eps', 'pdf', 'png'):
        p = os.path.splitext(out_path)[0] + '.' + ext
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print('saved', p)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=DEFAULT_RESULTS, help='dir with alignment_eval_{method}_32d_not_all8_streams.pkl')
    ap.add_argument('--out', default=os.path.join(HERE, 'figure3_crosssubj.eps'),
                    help='.eps path; the .pdf and .png are written next to it')
    a = ap.parse_args()
    print('results dir:', os.path.abspath(a.results))
    vae, srm, proc = (load_pairwise(a.results, m) for m in ('vae', 'srm', 'procrustes'))
    create_figure(vae, srm, proc, a.out)


if __name__ == '__main__':
    main()

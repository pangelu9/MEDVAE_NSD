"""Print Table 1 — cross-subject retrieval, top-k accuracy (%).

For each ordered pair of subjects (1, 2, 5, 7), every test image's latent in the query
subject is compared with the latents of all test images in the other subject; top-k
accuracy is the % of query images whose counterpart (the same image in the other subject)
is among the k nearest (evaluation/retrieval.py). The table shows the mean over the 12
ordered pairs. On the 128 not_all8 test images the gallery is all 128 images (one
deterministic pass), so chance is k/128.

Input (produced by the release pipeline — see README):
  <results>/alignment_eval_{vae,srm,procrustes}_32d_not_all8_streams.pkl   (evaluation/evaluate_methods.py)
  key retrieval[metric]['mean'][k] (in %; k = 1, 2, 3, 5, 10 stored, the table shows 1, 2, 5, 10)
  and retrieval[metric]['gallery_size'].
<results> defaults to $CCN_RESULTS_DIR, else evaluation/results.
One table is printed per stored distance metric: Euclidean (the paper's Table 1) and cosine.

Paper, Table 1 (%):
               top-1   top-2   top-5  top-10
  Chance         0.8     1.6     3.9     7.8
  Procrustes    18.4    25.6    40.9    54.1
  SRM           21.8    30.0    44.5    57.8
  MED-VAE       46.2    62.0    80.3    91.1
Rounded to one decimal, the bundled results give these values except SRM top-2:
29.948 (printed 29.95), 30.0 in the paper.

Usage:
  python render_table1.py [--results <dir>] [--out table1.csv]
"""
import os
import pickle
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.environ.get(
    'CCN_RESULTS_DIR', os.path.join(os.path.dirname(HERE), 'evaluation', 'results'))

TOP_K = [1, 2, 5, 10]
ROWS = [('procrustes', 'Procrustes'), ('srm', 'SRM'), ('vae', 'MED-VAE')]   # below the chance row, as in the paper
METRIC_LABELS = {'euclidean': "Euclidean distance (the paper's Table 1)", 'cosine': 'cosine distance'}


def load_retrieval(results_dir):
    """{method: results['retrieval']} ({distance metric: retrieval results}) from the eval pkls."""
    R = {}
    for m, _ in ROWS:
        path = os.path.join(results_dir, f'alignment_eval_{m}_32d_not_all8_streams.pkl')
        R[m] = pickle.load(open(path, 'rb')).get('retrieval')
        if not R[m]:
            raise SystemExit(f'{path}: no retrieval results')
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=DEFAULT_RESULTS, help='dir with alignment_eval_{method}_32d_not_all8_streams.pkl')
    ap.add_argument('--out', help='optional CSV of the printed tables')
    a = ap.parse_args()
    print('results dir:', os.path.abspath(a.results))
    R = load_retrieval(a.results)
    lines = ['metric,row,' + ','.join(f'top{k}' for k in TOP_K)]
    for metric in [mt for mt in R['vae'] if all(mt in R[m] for m in R)]:
        sizes = {R[m][metric]['gallery_size'] for m in R}
        if len(sizes) > 1:
            raise SystemExit(f'{metric}: the methods were evaluated with different gallery sizes {sorted(sizes)}')
        n = sizes.pop()
        rows = [('Chance', [100.0 * k / n for k in TOP_K])]   # random ranking: P(own image in top k) = k/n
        rows += [(label, [R[m][metric]['mean'][k] for k in TOP_K]) for m, label in ROWS]
        print(f'\nCross-subject retrieval, {METRIC_LABELS.get(metric, metric)}')
        print(f'top-k accuracy in %, mean over the {len(R["vae"][metric]["per_pair"])} ordered subject pairs; '
              f'gallery of {n} images (chance = k/{n})')
        print(f'{"":12}' + ''.join(f'{"top-" + str(k):>8}' for k in TOP_K))
        for label, vals in rows:
            print(f'{label:12}' + ''.join(f'{v:8.2f}' for v in vals))
            lines.append(f'{metric},{label},' + ','.join(f'{v:.4f}' for v in vals))
    if a.out:
        with open(a.out, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('\nwrote', os.path.abspath(a.out))


if __name__ == '__main__':
    main()

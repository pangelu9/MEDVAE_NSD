"""Generate Panel-A latent-space data for MED-VAE / SRM / Procrustes (self-contained).

For each method, encode/align the held-out VALIDATION images per subject, select a
label-balanced subset, reduce all subjects jointly to 2D with UMAP (n_components=2,
random_state=42 — identical to the original figure), and dump
  {encoder_idx: {'representations': (n,2), 'labels': (n,n_cat)}}  for Figure-2 Panel A.

  --method vae         : encode via the MED-VAE per-subject encoders (mu)
  --method srm|procrustes : apply the 872-fit alignment  (pca.transform - mean) @ W

All three reuse MED-VAE's own validation loader (medvae/load_data) built from the VAE
checkpoint's config, so the image set is identical across methods.

  sbatch ... visualise_latent.py --method vae        --vae_checkpoint <ckpt> --out <pkl>
  sbatch ... visualise_latent.py --method srm        --vae_checkpoint <ckpt> --alignment_model <pkl> --out <pkl>
  sbatch ... visualise_latent.py --method procrustes --vae_checkpoint <ckpt> --alignment_model <pkl> --out <pkl>
"""
import os
import sys
import pickle
import argparse

import numpy as np
import torch
import umap

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # medvae_release/
sys.path.insert(0, _REPO)                                  # for ccn_config
sys.path.insert(0, os.path.join(_REPO, 'medvae'))          # for pipeline / common_samples

from pipeline import load_data, create_vae_model          # noqa: E402
from common_samples import find_common_samples            # noqa: E402


# ---------- vendored: joint 2D UMAP (multiencoder_alignment_paradox.py:499) ----------
def apply_umap_reduction(encoder_data, target_dim=2):
    all_latents, sizes = [], {}
    for enc_idx, data in encoder_data.items():
        all_latents.append(data['representations'])
        sizes[enc_idx] = len(data['representations'])
    combined = np.vstack(all_latents)
    if combined.shape[1] <= target_dim:
        return encoder_data
    reduced = umap.UMAP(n_components=target_dim, random_state=42).fit_transform(combined)
    out, start = {}, 0
    for enc_idx in encoder_data:
        end = start + sizes[enc_idx]
        out[enc_idx] = {'representations': reduced[start:end],
                        'labels': encoder_data[enc_idx]['labels']}
        start = end
    print(f"UMAP reduction: {combined.shape[1]}D -> {target_dim}D")
    return out


# ---------- vendored greedy label-balanced selection (select_images.py:121),
# ---------- vectorized: identical objective, scored over all candidates per step ----------
def _iterative_balanced_selection(all_data, n_images, n_classes):
    print(f"Iterative balanced selection for {n_images} images...")
    L = np.array([d['labels'] for d in all_data], dtype=float)   # (N, C)
    N = L.shape[0]
    class_counts = np.zeros(n_classes)
    available = np.ones(N, dtype=bool)
    chosen = []
    for _ in range(min(n_images, N)):
        new_std = (L + class_counts).std(axis=1)                 # std(class_counts + label) per candidate
        underrep = (L * (1.0 / (class_counts + 1))).sum(axis=1)
        scores = np.where(available, new_std - 0.1 * underrep, np.inf)
        b = int(np.argmin(scores))                               # first min (ascending index), greedy
        chosen.append(b); available[b] = False; class_counts += L[b]
    return [all_data[i] for i in chosen]


def _to_encoder_data(selected):
    enc_idx_set = sorted({e for s in selected for e in s['representations']})
    enc = {e: {'representations': [], 'labels': []} for e in enc_idx_set}
    for s in selected:
        for e in enc_idx_set:
            if e in s['representations']:
                enc[e]['representations'].append(s['representations'][e])
                enc[e]['labels'].append(s['labels'])
    for e in enc_idx_set:
        enc[e]['representations'] = np.array(enc[e]['representations'])
        enc[e]['labels'] = np.array(enc[e]['labels'])
        print(f"  Encoder {e}: {enc[e]['representations'].shape}")
    return enc


def extract_balanced(loader, n_enc, batch_encode, n_images):
    """batch_encode(X_subj (n,V) numpy, enc_idx) -> (n, latent_dim) numpy.
    fMRI is gathered per subject (respecting masks) and encoded/aligned in ONE batched call."""
    labels_list, rows, idxs = [], {e: [] for e in range(n_enc)}, {e: [] for e in range(n_enc)}
    s = 0
    with torch.no_grad():
        for encoder_inputs, labels, _, masks in loader:
            lab = labels.cpu().numpy()
            for i in range(labels.size(0)):
                labels_list.append(lab[i])
                for e in range(n_enc):
                    if masks[i, e]:
                        rows[e].append(encoder_inputs[e][i].cpu().numpy())
                        idxs[e].append(s)
                s += 1
    N = len(labels_list)
    print(f"Collected {N} validation samples; batch-encoding per subject...")
    reps = [dict() for _ in range(N)]
    for e in range(n_enc):
        if not rows[e]:
            continue
        Z = batch_encode(np.stack(rows[e]), e)                  # one call for the whole subject
        for j, sid in enumerate(idxs[e]):
            reps[sid][e] = Z[j]
    all_data = [{'labels': labels_list[k], 'representations': reps[k]} for k in range(N)]
    selected = _iterative_balanced_selection(all_data, min(n_images, N), all_data[0]['labels'].shape[0])
    return _to_encoder_data(selected)


def build_val_loader(ckpt, device):
    """Reconstruct the MED-VAE validation loader from the checkpoint's training config."""
    config = dict(ckpt.get('config', {}))
    for k in ('fmri_common', 'nn_common', 'labels_common', 'device'):
        config.pop(k, None)
    args = argparse.Namespace(**config)
    args.device, args.no_cuda = device, (device.type != 'cuda')
    print(f"loader config: dataset={args.dataset} latent_dim={args.latent_dim} "
          f"hybrid_vae={getattr(args, 'hybrid_vae', None)}")
    fc, nc, lc, _ = find_common_samples(args)
    args.fmri_common, args.nn_common, args.labels_common = fc, nc, lc
    kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}
    _, test_loader, *_ = load_data(args, kwargs, gradual_intro_enc=False)
    return args, test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['vae', 'srm', 'procrustes'], default='vae')
    ap.add_argument('--vae_checkpoint', required=True, help='MED-VAE .pt (also supplies the val-loader config)')
    ap.add_argument('--alignment_model', help='SRM/Procrustes alignment .pkl (for --method srm|procrustes)')
    ap.add_argument('--n_images', type=int, default=3000)
    ap.add_argument('--out', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); np.random.seed(42)

    ckpt = torch.load(cli.vae_checkpoint, map_location=device, weights_only=False)
    args, test_loader = build_val_loader(ckpt, device)

    if cli.method == 'vae':
        model = create_vae_model(args, args.input_dim, args.output_dim).to(device)
        model.load_state_dict(ckpt['model_state_dict']); model.eval()
        n_enc = len(model.encoders) - 1                      # exclude NN encoder
        def batch_encode(X, e):                              # (n,V) numpy -> (n,latent) on GPU, chunked
            out = []
            with torch.no_grad():
                for k in range(0, len(X), 4096):
                    xb = torch.from_numpy(X[k:k + 4096]).float().to(device)
                    mu, _ = model.encoders[e](xb)
                    out.append(mu.detach().cpu().numpy())
            return np.concatenate(out, axis=0)
    else:
        am = pickle.load(open(cli.alignment_model, 'rb'))
        W, pca, means = am['W_list'], am['pca_models'], am['training_means']
        n_enc = am['n_subjects']
        print(f"alignment: {am['method_name']}  n_pca={am['n_pca_features']} -> {am['n_components']}")
        def batch_encode(X, e):                              # (pca.transform - mean) @ W, batched over all rows
            return (pca[e].transform(X.astype(np.float64)) - means[e]) @ W[e]

    encoder_data = extract_balanced(test_loader, n_enc, batch_encode, cli.n_images)
    encoder_data_2d = apply_umap_reduction(encoder_data, target_dim=2)

    os.makedirs(os.path.dirname(cli.out), exist_ok=True)
    with open(cli.out, 'wb') as f:
        pickle.dump(encoder_data_2d, f)
    print("Saved", cli.out)
    for k, v in encoder_data_2d.items():
        print(f"  encoder {k}: {v['representations'].shape}  labels {v['labels'].shape}")


if __name__ == '__main__':
    main()

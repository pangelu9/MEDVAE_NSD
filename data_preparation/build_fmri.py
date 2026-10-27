"""Step 3/4 -- build the per-subject fMRI files from the Algonauts 2023 data.

For each subject, the Algonauts responses (already z-scored within each NSD session and
averaged over the three image repeats) of the training AND test split are:

  1. concatenated left hemisphere then right hemisphere (challenge-space vertices),
  2. restricted to the ``streams`` ROIs midventral, midlateral, ventral and lateral
     (labels 2, 3, 5, 6 of ``roi_masks/{lh,rh}.streams_challenge_space.npy``),
     ~20,730 vertices per subject,
  3. placed in a (70502, n_vertices) float32 array in the row order of ``nsd_ids.npy``
     (step 2), with NaN for images the subject did not see.

No further normalisation is applied. Output: ``fmri_subject{01..08}.npz`` with a single
array ``fmri_data``.

    python build_fmri.py --algonauts_dir /path/to/algonauts_2023_challenge_data \
        --nsd_ids $CCN_DATA_ROOT/nsd/data/nsd_ids.npy --out_dir $CCN_DATA_ROOT/nsd/data
"""
import argparse
import os

import numpy as np

SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08"]
STREAMS_ROIS = [2, 3, 5, 6]          # midventral, midlateral, ventral, lateral
SPLITS = [("training_split", "training_images", "training_fmri", "training"),
          ("test_split", "test_images", "test_fmri", "test")]


def nsd_id_from_filename(name):
    return str(int(name.split("_nsd-")[1].split(".")[0]))


def build_subject(algonauts_dir, sid, row_of, n_rows, rois):
    subj_dir = os.path.join(algonauts_dir, f"subj{sid}")
    roi = np.concatenate([
        np.load(os.path.join(subj_dir, "roi_masks", f"{h}.streams_challenge_space.npy"))
        for h in ("lh", "rh")])
    keep = np.isin(roi, rois)
    data = np.full((n_rows, int(keep.sum())), np.nan, dtype=np.float32)
    n_filled = 0
    for split, img_dir, fmri_dir, prefix in SPLITS:
        lh = np.load(os.path.join(subj_dir, split, fmri_dir, f"lh_{prefix}_fmri.npy"), mmap_mode="r")
        rh = np.load(os.path.join(subj_dir, split, fmri_dir, f"rh_{prefix}_fmri.npy"), mmap_mode="r")
        if lh.shape[1] + rh.shape[1] != len(keep):
            raise ValueError(f"subj{sid} {split}: {lh.shape[1]}+{rh.shape[1]} vertices "
                             f"!= {len(keep)} ROI labels")
        # Algonauts fMRI rows follow the sorted image file names (train-0001, train-0002, ...).
        names = sorted(n for n in os.listdir(os.path.join(subj_dir, split, img_dir)) if "_nsd-" in n)
        if len(names) != lh.shape[0] or len(names) != rh.shape[0]:
            raise ValueError(f"subj{sid} {split}: {len(names)} images vs {lh.shape[0]} fMRI rows")
        for k, name in enumerate(names):
            row = row_of.get(nsd_id_from_filename(name))
            if row is None:                      # image without labels: not in the index
                continue
            data[row] = np.concatenate([lh[k], rh[k]])[keep]
            n_filled += 1
    return data, n_filled


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--algonauts_dir", required=True,
                    help="folder containing subj01..subj08 of the Algonauts 2023 data")
    ap.add_argument("--nsd_ids", required=True, help="nsd_ids.npy from build_index_and_labels.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS)
    ap.add_argument("--rois", type=int, nargs="+", default=STREAMS_ROIS,
                    help="streams ROI labels to keep (default: 2 3 5 6)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    nsd_ids = np.load(args.nsd_ids)
    row_of = {str(i): r for r, i in enumerate(nsd_ids)}
    for sid in args.subjects:
        data, n_filled = build_subject(args.algonauts_dir, sid, row_of, len(nsd_ids), args.rois)
        out = os.path.join(args.out_dir, f"fmri_subject{sid}.npz")
        np.savez_compressed(out, fmri_data=data)
        print(f"[fmri] subj{sid}: {data.shape}, {n_filled} images with data -> {out}", flush=True)
        del data


if __name__ == "__main__":
    main()

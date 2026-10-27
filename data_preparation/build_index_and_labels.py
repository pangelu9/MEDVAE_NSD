"""Step 2/4 -- define the image index shared by all data files, and the category labels.

Every data file of the pipeline (per-subject fMRI, ANN features, labels) has one row
per NSD image, in ONE common order. That order is defined here:

  * the images are all NSD images that any of the 8 subjects saw in the Algonauts 2023
    training OR test split (read from the image file names ``*_nsd-NNNNN.png``),
  * restricted to images that have COCO labels in NSD-Flat (step 1) -- 64 images
    without any annotation are dropped, leaving 70,502 images,
  * sorted as STRINGS, i.e. '0', '1', '10', '100', '1000', '10000', ... (this is the
    order of the paper's files; it is not numeric order).

Outputs (in --out_dir):
  nsd_ids.npy             (70502,) unicode NSD image ids (0-based), the row order
  category_labels.npy     (70502, 12) float32 multi-hot COCO super-categories
  category_labels_names.json  the 12 column names (alphabetical)

    python build_index_and_labels.py --algonauts_dir /path/to/algonauts_2023_challenge_data \
        --labels_json nsd_flat_labels.json --out_dir $CCN_DATA_ROOT/nsd/data
"""
import argparse
import json
import os

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08"]
IMAGE_DIRS = [("training_split", "training_images"), ("test_split", "test_images")]


def nsd_id_from_filename(name):
    """'train-0001_nsd-00013.png' -> '13' (0-based NSD id, no zero padding)."""
    return str(int(name.split("_nsd-")[1].split(".")[0]))


def algonauts_nsd_ids(algonauts_dir, subjects=SUBJECTS):
    """All NSD ids seen by any subject in the Algonauts train or test split."""
    ids = set()
    for sid in subjects:
        for split, img_dir in IMAGE_DIRS:
            d = os.path.join(algonauts_dir, f"subj{sid}", split, img_dir)
            if not os.path.isdir(d):
                raise FileNotFoundError(f"missing {d} (unzip both train_data and test_data)")
            ids.update(nsd_id_from_filename(n) for n in os.listdir(d) if "_nsd-" in n)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--algonauts_dir", required=True,
                    help="folder containing subj01..subj08 of the Algonauts 2023 data")
    ap.add_argument("--labels_json", required=True, help="output of fetch_labels.py")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.labels_json) as f:
        labels = json.load(f)
    seen = algonauts_nsd_ids(args.algonauts_dir)
    kept = [i for i in seen if i in labels]
    # np.unique on strings sorts lexicographically: this IS the canonical row order.
    nsd_ids = np.unique(np.array(kept))
    print(f"[index] {len(seen)} images in Algonauts train+test, "
          f"{len(seen) - len(kept)} without labels dropped -> {len(nsd_ids)} rows")

    mlb = MultiLabelBinarizer()
    multi_hot = mlb.fit_transform([labels[i]["supercategory"] for i in nsd_ids]).astype(np.float32)

    np.save(os.path.join(args.out_dir, "nsd_ids.npy"), nsd_ids)
    np.save(os.path.join(args.out_dir, "category_labels.npy"), multi_hot)
    with open(os.path.join(args.out_dir, "category_labels_names.json"), "w") as f:
        json.dump([str(c) for c in mlb.classes_], f, indent=1)
    print(f"[labels] {multi_hot.shape} multi-hot super-categories: {list(mlb.classes_)}")


if __name__ == "__main__":
    main()

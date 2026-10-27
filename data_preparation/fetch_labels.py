"""Step 1/4 -- fetch the COCO object labels of every NSD image from NSD-Flat.

NSD-Flat (HuggingFace dataset ``clane9/NSD-Flat``) carries, for every NSD trial, the
COCO annotations of the image shown. Only the ``nsd_id`` and ``objects`` columns are
read (column projection over the remote Parquet files), so the ~23 GB of images and
activity maps in the dataset are NOT downloaded.

Output: a JSON file mapping each NSD image id (0-based, as a string) to the set of
COCO categories and super-categories annotated in it::

    {"13": {"category": ["dog", "person"], "supercategory": ["animal", "person"]}, ...}

Images whose annotation list is empty are omitted (they are later dropped from the
dataset). Rows of the train split are read first and the test split second, exactly
as the paper's mapping was built (both splits annotate an image identically).

    python fetch_labels.py --out nsd_flat_labels.json
"""
import argparse
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO_ID = "clane9/NSD-Flat"
# The revision the paper's labels were built from (the dataset has not changed since).
DEFAULT_REVISION = "350bf18ab82c98e58e57fe91657ef9cc5a74272c"


def label_columns(table):
    """Per-row (categories, supercategories), whether the reader returned the 'objects'
    struct (pruned to the selected fields) or the fields as separate columns."""
    names = table.column_names
    if "objects" in names:
        objects = table.column("objects").to_pylist()
        return ([o.get("category") if o else None for o in objects],
                [o.get("supercategory") if o else None for o in objects])
    cat = [n for n in names if n.split(".")[-1] == "category"]
    sup = [n for n in names if n.split(".")[-1] == "supercategory"]
    if len(cat) != 1 or len(sup) != 1:
        raise KeyError(f"label fields not found among the columns read: {names}")
    return table.column(cat[0]).to_pylist(), table.column(sup[0]).to_pylist()


def read_split(fs, root, split):
    """Yield (nsd_id, categories, supercategories) for every row of one split."""
    files = sorted(fs.glob(f"{root}/data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet files under {root}/data")
    for path in files:
        print(f"  reading {path.split('/')[-1]}", flush=True)
        # only the two label fields of the 'objects' struct (skips images, boxes, masks)
        table = pq.ParquetFile(path, filesystem=fs).read(
            columns=["nsd_id", "objects.category", "objects.supercategory"])
        cats, supercats = label_columns(table)
        for nsd_id, c, s in zip(table.column("nsd_id").to_pylist(), cats, supercats):
            if c:
                yield nsd_id, c, s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output JSON (e.g. nsd_flat_labels.json)")
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help="NSD-Flat git revision (default: the one used for the paper)")
    args = ap.parse_args()

    fs = HfFileSystem()
    root = f"datasets/{REPO_ID}@{args.revision}"
    labels = {}
    for split in ("train", "test"):
        print(f"[fetch_labels] {REPO_ID} split={split}")
        for nsd_id, cats, supercats in read_split(fs, root, split):
            labels[str(int(nsd_id))] = {"category": sorted(set(cats)),
                                        "supercategory": sorted(set(supercats))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(labels, f)
    print(f"[fetch_labels] wrote {len(labels)} labelled NSD images -> {args.out}")


if __name__ == "__main__":
    main()

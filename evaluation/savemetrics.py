
from typing import List, Optional
import pathlib, pickle
import time 
import numpy as np

def save_alignment_metrics(
    comp_corr: float,
    rsa_euclidean: float,
    rsa_pearson: float,
    comp_corr_per_component: Optional[List[float]] = None,
    rsa_matrix: Optional[np.ndarray] = None,
    tag: Optional[str] = None,
    subset_tag: Optional[str] = None,
    out_dir: str = "results",
    rsa_pearson_centered: Optional[float] = None,
    rsa_matrix_centered: Optional[np.ndarray] = None,
) -> None:
    """
    Save alignment metrics with FULL distributions.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    stamp = tag if tag is not None else str(int(time.time()))
    subset_str = f"_{subset_tag}" if subset_tag else ""

    results = {
        'comp_corr_mean': comp_corr,
        'comp_corr': comp_corr,  # For backward compatibility
        'rsa_euclidean': rsa_euclidean,
        'rsa_pearson': rsa_pearson,
        'rsa': rsa_pearson,  # Default to Pearson
        'comp_corr_per_component': comp_corr_per_component,
        'rsa_matrix': rsa_matrix,            # raw (headline)
        # Centered (legacy) RSA variant, kept for reference (None on legacy callers).
        'rsa_pearson_centered': rsa_pearson_centered,
        'rsa_matrix_centered': rsa_matrix_centered,
    }
    
    with open(out_path / f"alignment_metrics_{stamp}{subset_str}.pkl", "wb") as f:
        pickle.dump(results, f)
    
    print(f" Saved alignment metrics to: {out_path / f'alignment_metrics_{stamp}{subset_str}.pkl'}")
    
    # ← FIX: Use "is not None" instead of implicit truthiness
    if comp_corr_per_component is not None:
        print(f"   Comp corr: mean={comp_corr:.4f}, per-component={len(comp_corr_per_component)} values")
    else:
        print(f"   Comp corr: mean={comp_corr:.4f}, per-component=None")
    
    print(f"   RSA Euclidean: {rsa_euclidean:.4f}, Pearson: {rsa_pearson:.4f}")
    
    # ← FIX: Use "is not None" here too
    if rsa_matrix is not None:
        print(f"   RSA matrix shape: {rsa_matrix.shape}")
    else:
        print(f"   RSA matrix: None")

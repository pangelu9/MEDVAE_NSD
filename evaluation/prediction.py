import numpy as np
from typing import Dict, List, Optional, Any
import pickle
import pathlib

"""
Updated pipeline for analyzing images seen by ≥2 subjects (pairwise common samples)
with proper centering using training means.

"""





def run_pairwise_analysis_pipeline(
    pairwise_data_dict: Dict,
    subject_sample_lists: List[List[int]],
    W_list: List[np.ndarray],
    pca_models: List,
    training_means: List[np.ndarray],
    method_name: str = "srm_pairwise",
    min_shared_images: int = 5,
    compute_original_space: bool = True,
    exclude_subjects: List[int] = None,
    output_dir: str = "results/other",
    compute_reconstruction: bool = True  
):
    """
    Complete pipeline for pairwise common sample analysis.
    
    Parameters:
    -----------
    exclude_subjects : list of int
        Subject indices to exclude from analysis (e.g., [2, 5] for subjects 3 and 6)
    """
    from metrics import compute_fmri_recon_metrics
    
    n_subjects = len(W_list)
    
    if exclude_subjects is None:
        exclude_subjects = []
    
    # Subjects to include
    include_subjects = [i for i in range(n_subjects) if i not in exclude_subjects]
    
    print(f"\n{'='*70}")
    print(f"PAIRWISE ANALYSIS: {method_name}")
    print(f"{'='*70}")
    print(f"Excluding subjects: {[s+1 for s in exclude_subjects]}")
    print(f"Including subjects: {[s+1 for s in include_subjects]}")
    
    # Storage for results
    R_pca = np.full((n_subjects, n_subjects), np.nan)
    R_original = np.full((n_subjects, n_subjects), np.nan)
    n_images_per_pair = np.zeros((n_subjects, n_subjects), dtype=int)
    pairwise_correlations_pca = {}
    pairwise_correlations_original = {}
    
    # Ensure training means have correct shape
    training_means = [m.reshape(1, -1) if m.ndim == 1 else m for m in training_means]

    reconstruction_per_subject = {
        'voxel_corr': [None] * n_subjects,
        'r2': [None] * n_subjects,
        'n_samples': [0] * n_subjects
    }
    
    print(f"\n{'='*70}")
    print(f"PAIRWISE CROSS-SUBJECT PREDICTION: {method_name}")
    print(f"{'='*70}")
    print(f"Using training means for centering: ")
    print(f"Min shared images per pair: {min_shared_images}")
    if exclude_subjects:
        print(f"Excluding subjects: {[s+1 for s in exclude_subjects]}")

    if compute_reconstruction:
        print(f"\n{'='*70}")
        print(f"RECONSTRUCTION QUALITY (128 PAIRWISE SAMPLES)")
        print(f"{'='*70}")
        
        for subj in range(n_subjects):
            if exclude_subjects and subj in exclude_subjects:
                print(f"  Subject {subj+1}: EXCLUDED")
                continue
            
            # Get all samples this subject saw
            subj_samples = subject_sample_lists[subj]
            
            if len(subj_samples) < min_shared_images:
                print(f"  Subject {subj+1}: Only {len(subj_samples)} samples, skipping")
                continue
            
            # Extract original voxel data
            X_voxel = np.stack([pairwise_data_dict[idx]['data'][subj] 
                               for idx in subj_samples])
            
            # Transform to PCA space
            X_pca = pca_models[subj].transform(X_voxel)
            
            # Center using training means
            X_centered = X_pca - training_means[subj]
            
            # Self-reconstruction: X → aligned → reconstructed
            X_aligned = X_centered @ W_list[subj]
            X_recon_pca = X_aligned @ W_list[subj].T
            
            # Back to voxel space
            X_recon_voxel = pca_models[subj].inverse_transform(X_recon_pca)
            
            # Compute metrics
            metrics = compute_fmri_recon_metrics(X_voxel, X_recon_voxel, voxel_thresh=0.1)
            
            reconstruction_per_subject['voxel_corr'][subj] = metrics['voxel_correlation_mean']
            reconstruction_per_subject['r2'][subj] = metrics['r2']
            reconstruction_per_subject['n_samples'][subj] = len(subj_samples)
            
            print(f"  Subject {subj+1}: voxel_corr={metrics['voxel_correlation_mean']:.4f}, "
                  f"R²={metrics['r2']:.4f} ({len(subj_samples)} samples)")
        
        # Compute mean across valid subjects
        valid_voxel_corr = [v for v in reconstruction_per_subject['voxel_corr'] if v is not None]
        valid_r2 = [v for v in reconstruction_per_subject['r2'] if v is not None]
        
        print(f"\n  Mean voxel correlation: {np.mean(valid_voxel_corr):.4f}")
        print(f"  Mean R²: {np.mean(valid_r2):.4f}")
    
    # ========================================================================
    # 1. CROSS-SUBJECT PREDICTION
    # ========================================================================
    print("\n" + "="*70)
    print("CROSS-SUBJECT PREDICTION")
    print("="*70)
    
    for i in include_subjects:
        for j in include_subjects:
            # Find samples both subjects saw
            samples_i = set(subject_sample_lists[i])
            samples_j = set(subject_sample_lists[j])
            shared_sample_ids = sorted(list(samples_i & samples_j))
            
            n_shared = len(shared_sample_ids)
            n_images_per_pair[i, j] = n_shared
            
            if n_shared < min_shared_images:
                if i != j:
                    print(f"  S{i+1}↔S{j+1}: Only {n_shared} shared samples (< {min_shared_images}), skipping")
                continue
            
            # Extract original voxel data
            X_i_voxel = np.stack([pairwise_data_dict[idx]['data'][i] 
                                  for idx in shared_sample_ids])
            X_j_voxel = np.stack([pairwise_data_dict[idx]['data'][j] 
                                  for idx in shared_sample_ids])
            
            # Transform to PCA space
            X_i_pca = pca_models[i].transform(X_i_voxel)
            X_j_pca = pca_models[j].transform(X_j_voxel)
            
            # Center using TRAINING means
            X_i_centered = X_i_pca - training_means[i]
            
            # Predict j from i
            if i == j:
                X_j_pred_pca = X_i_centered @ W_list[i] @ W_list[i].T
            else:
                X_j_pred_pca = X_i_centered @ W_list[i] @ W_list[j].T
            
            # Evaluate in PCA space
            metrics_pca = compute_fmri_recon_metrics(X_j_pca, X_j_pred_pca, voxel_thresh=0.1)
            R_pca[i, j] = metrics_pca['voxel_correlation_mean']
            pairwise_correlations_pca[(i, j)] = metrics_pca['voxel_correlation_list']
            
            # Evaluate in original voxel space (optional)
            if compute_original_space:
                X_j_pred_voxel = pca_models[j].inverse_transform(X_j_pred_pca)
                metrics_original = compute_fmri_recon_metrics(
                    X_j_voxel, X_j_pred_voxel, voxel_thresh=0.1
                )
                R_original[i, j] = metrics_original['voxel_correlation_mean']
                pairwise_correlations_original[(i, j)] = metrics_original['voxel_correlation_list']
            
            # Print progress
            if i == j:
                space_str = f"PCA: {R_pca[i,j]:.4f}"
                if compute_original_space:
                    space_str += f", Voxel: {R_original[i,j]:.4f}"
                print(f"  S{i+1}→S{j+1} (self): {space_str} ({n_shared} images)")
            elif i < 2:
                space_str = f"PCA: {R_pca[i,j]:.4f}"
                if compute_original_space:
                    space_str += f", Voxel: {R_original[i,j]:.4f}"
                print(f"  S{i+1}→S{j+1}: {space_str} ({n_shared} images)")
    
    # ========================================================================
    # 2. ALIGNMENT METRICS (on common pairwise samples)
    # ========================================================================
    print("\n" + "="*70)
    print("ALIGNMENT METRICS (Component Corr & RSA)")
    print("="*70)
    
    # Find images seen by ALL included subjects
    common_samples = set(subject_sample_lists[include_subjects[0]])
    for subj_idx in include_subjects[1:]:
        common_samples = common_samples & set(subject_sample_lists[subj_idx])
    common_samples = sorted(list(common_samples))
    
    print(f"Common samples across included subjects: {len(common_samples)}")
    
    if len(common_samples) >= min_shared_images:
        # Extract and align common samples
        aligned_pairwise = []
        
        for subj_idx in include_subjects:
            # Get data for common samples
            X_voxel = np.stack([pairwise_data_dict[idx]['data'][subj_idx] 
                               for idx in common_samples])
            
            # Transform to PCA and align
            X_pca = pca_models[subj_idx].transform(X_voxel)
            X_centered = X_pca - training_means[subj_idx]
            X_aligned = X_centered @ W_list[subj_idx]
            
            aligned_pairwise.append(X_aligned)
        
        # Compute alignment metrics on these samples
        n_common = len(common_samples)
        valid_idx_list = [np.arange(n_common) for _ in include_subjects]
        
        from metrics import cross_subj_metrics
        
        metrics = cross_subj_metrics(
        latents_clean=aligned_pairwise,
        valid_idx_list=valid_idx_list,
        method=method_name
        )
        
        comp_corr = metrics['avg_comp_corr']
        rsa = metrics['avg_rsa_pearson']
        
        # Get full distributions
        comp_corr_per_comp = metrics.get('per_component_corr', None)
        rsa_matrix = metrics.get('rsa_matrix_pearson', None)
        
        # Save with full distributions
        # Save alignment metrics
        from metrics import save_alignment_metrics
        save_alignment_metrics(
            comp_corr=comp_corr,
            rsa_euclidean=metrics.get('avg_rsa_euclidean', rsa),
            rsa_pearson=rsa,
            comp_corr_per_component=comp_corr_per_comp,  # ← ADD
            rsa_matrix=rsa_matrix,  # ← ADD
            tag=method_name,
            out_dir=output_dir
        )
        
    else:
        print(f"  Not enough common samples ({len(common_samples)} < {min_shared_images})")
        comp_corr = np.nan
        rsa = np.nan
    
    # ========================================================================
    # Summary statistics
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    # Create mask for included subjects
    mask = np.zeros((n_subjects, n_subjects), dtype=bool)
    for i in include_subjects:
        for j in include_subjects:
            mask[i, j] = True
    
    # PCA space
    diag_pca = np.nanmean(R_pca[np.diag_indices(n_subjects)][include_subjects])
    offdiag_mask = mask & ~np.eye(n_subjects, dtype=bool)
    offdiag_pca = np.nanmean(R_pca[offdiag_mask])
    
    print(f"\nCross-Subject Prediction (PCA Space):")
    print(f"  Diagonal (self-recon) mean:     {diag_pca:.4f}")
    print(f"  Off-diagonal (cross-pred) mean: {offdiag_pca:.4f}")
    print(f"  Valid pairs: {np.sum(~np.isnan(R_pca[offdiag_mask]))}")
    
    # Original space
    if compute_original_space:
        diag_orig = np.nanmean(R_original[np.diag_indices(n_subjects)][include_subjects])
        offdiag_orig = np.nanmean(R_original[offdiag_mask])
        print(f"\nCross-Subject Prediction (Original Voxel Space):")
        print(f"  Diagonal (self-recon) mean:     {diag_orig:.4f}")
        print(f"  Off-diagonal (cross-pred) mean: {offdiag_orig:.4f}")
    
    print(f"\nAlignment Metrics:")
    print(f"  Component-wise correlation: {comp_corr:.4f}")
    print(f"  RSA: {rsa:.4f}")
    
    print(f"\nPrediction Matrix (PCA space):")
    print(R_pca)
    
    if compute_original_space:
        print(f"\nPrediction Matrix (Original voxel space):")
        print(R_original)
    
    print(f"\nSample counts per pair:")
    print(n_images_per_pair)
    print(f"{'='*70}\n")
    
    # ========================================================================
    # Save results
    # ========================================================================
    results = {
        'prediction_matrix_pca': R_pca,
        'prediction_matrix_original': R_original if compute_original_space else None,
        'n_images_matrix': n_images_per_pair,
        'pairwise_correlations_pca': pairwise_correlations_pca,
        'pairwise_correlations_original': pairwise_correlations_original if compute_original_space else None,
        'diagonal_mean_pca': diag_pca,
        'offdiagonal_mean_pca': offdiag_pca,
        'diagonal_mean_original': diag_orig if compute_original_space else None,
        'offdiagonal_mean_original': offdiag_orig if compute_original_space else None,
        'alignment_comp_corr': comp_corr,
        'alignment_rsa': rsa,
        'min_shared_images': min_shared_images,
        'exclude_subjects': exclude_subjects,
        'method_name': method_name,
        'reconstruction_per_subject': reconstruction_per_subject if compute_reconstruction else None,
        'reconstruction_mean_voxel_corr': np.mean(valid_voxel_corr) if compute_reconstruction else None,
        'reconstruction_mean_r2': np.mean(valid_r2) if compute_reconstruction else None,
    }
    
    # Save prediction matrices
    save_pairwise_results(results, method_name, out_dir=output_dir) 
    
    return results


def save_pairwise_results(results: Dict, tag: str, out_dir: str = "results/other"):  
    """Save pairwise analysis results."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    
    filename = out_path / f"pairwise_prediction_{tag}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    
    print(f" Saved pairwise results to: {filename}")




def evaluate_cross_subject_prediction(
    aligned_data: List[np.ndarray],
    W_list: List[np.ndarray],
    original_data: List[np.ndarray],
    method_name: str,
    training_means: List[np.ndarray],  # ← CRITICAL NEW PARAMETER
    pca_models: Optional[List] = None,
    data_before_pca: Optional[List[np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Compute cross-subject prediction matrix R[i,j] = voxel correlation 
    when transforming subject i to predict subject j.
    
    CRITICAL FIX: Now requires training_means parameter to ensure consistent
    centering between training and testing.
    
    Parameters:
    -----------
    aligned_data : list of (n_samples, n_components)
        Aligned representations in shared space
    W_list : list of (n_voxels_pca, n_components)
        Transformation matrices for each subject
    original_data : list of (n_samples, n_voxels_pca)
        PCA-reduced TEST data (what needs to be aligned)
    method_name : str
        Name for printing
    training_means : list of (n_voxels_pca,) arrays **REQUIRED**
        Mean of the TRAINING data (872 images) for each subject IN PCA SPACE.
        These are computed from the data used to train SRM/Procrustes.
    pca_models : list of PCA models (optional)
        If provided, transform back to original voxel space
    data_before_pca : list of (n_samples, n_voxels_original) (optional)
        Original voxel data before PCA
        
    Returns:
    --------
    dict with:
        - prediction_matrix : (n_subjects, n_subjects) array
        - per_voxel_correlations : dict
        - diagonal_mean : float
        - offdiagonal_mean : float
        - space : 'pca' or 'original'
    """
    from metrics import compute_fmri_recon_metrics
    
    n_subjects = len(aligned_data)
    R = np.zeros((n_subjects, n_subjects))
    per_voxel_corrs = {}
    
    # Determine if we're working in original or PCA space
    use_original_space = (data_before_pca is not None and pca_models is not None)
    
    print(f"\n{'='*70}")
    if use_original_space:
        print(f"CROSS-SUBJECT PREDICTION IN ORIGINAL VOXEL SPACE: {method_name}")
    else:
        print(f"CROSS-SUBJECT PREDICTION IN PCA SPACE: {method_name}")
    print(f"{'='*70}")
    
    # Ensure training means have correct shape (n_voxels,) or (1, n_voxels)
    training_means = [m.reshape(1, -1) if m.ndim == 1 else m for m in training_means]
    print(f" Using provided training means for centering")
    
    for i in range(n_subjects):
        # CRITICAL: Center subject i's TEST data using TRAINING mean (in PCA space)
        X_i_centered = original_data[i] - training_means[i]
        
        for j in range(n_subjects):
            if i == j:
                # Self-reconstruction (diagonal)
                X_j_pred_pca = aligned_data[i] @ W_list[i].T  # In PCA space
            else:
                # Cross-subject prediction: i → j (in PCA space)
                X_j_pred_pca = X_i_centered @ W_list[i] @ W_list[j].T
            
            # Transform to original space if requested
            if use_original_space:
                # Ground truth in original voxel space
                X_j_actual = data_before_pca[j]
                
                # Transform prediction back to original voxel space
                X_j_pred = pca_models[j].inverse_transform(X_j_pred_pca)
                
            else:
                # Stay in PCA space
                X_j_actual = original_data[j]
                X_j_pred = X_j_pred_pca
            
            # Compute voxel-wise correlation
            metrics = compute_fmri_recon_metrics(X_j_actual, X_j_pred, voxel_thresh=0.1)
            R[i, j] = metrics['voxel_correlation_mean']
            per_voxel_corrs[(i, j)] = metrics['voxel_correlation_list']
            
            if (i < 2) or (i == j):  # Print diagonal and first 2 rows
                print(f"  Transform S{i+1}→S{j+1}: r_voxel = {R[i, j]:.4f} "
                      f"({len(metrics['voxel_correlation_list'])} voxels)")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Space: {'Original (~20k voxels)' if use_original_space else 'PCA space'}")
    print(f"  Diagonal (self-recon) mean:  {np.diag(R).mean():.4f}")

    offdiagonal_all_mean = np.nanmean(R[~np.eye(n_subjects, dtype=bool)])
    print(f"  Off-diagonal (cross-pred) mean: {offdiagonal_all_mean:.4f}")
    print(f"\nPrediction Matrix R:")
    print(R)
    
    return {
        'prediction_matrix': R,
        'per_voxel_correlations': per_voxel_corrs,
        'diagonal_mean': np.diag(R).mean(),
        'offdiagonal_mean': offdiagonal_all_mean,
        'space': 'original' if use_original_space else 'pca'
    }


def compute_training_means_pca(data_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compute the mean of training data in PCA space.
    
    This should be called on the 872 common images used for SRM/Procrustes training.
    The resulting means should be saved and used for all subsequent predictions.
    
    Parameters:
    -----------
    data_list : list of (n_train_samples, n_pca_features) arrays
        The 872 PCA-reduced images used for training
        
    Returns:
    --------
    training_means : list of (n_pca_features,) arrays
        Mean for each subject in PCA space
    """
    training_means = [data.mean(axis=0) for data in data_list]
    
    print(f"\nComputed training means:")
    for i, mean in enumerate(training_means):
        print(f"  Subject {i+1}: shape {mean.shape}, norm={np.linalg.norm(mean):.4f}")
    
    return training_means


def save_training_means(training_means: List[np.ndarray], 
                       tag: str,
                       out_dir: str = "results"):
    """Save training means for later use."""
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    
    filename = out_path / f"training_means_{tag}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(training_means, f)
    
    print(f" Saved training means to: {filename}")


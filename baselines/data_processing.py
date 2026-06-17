# Vendored verbatim from ../baselines_old/data_processing.py so baselines/ is
# fully self-contained for the paper release (loaded locally by the fit_* drivers).
import numpy as np
from tqdm import tqdm
import os

def extract_full_subject_data(data_loader, n_subjects=8):
    """
    Extract ALL available data for each subject (not just common samples).
    This is used to fit PCA on maximum data for each subject.
    """
    print("\nExtracting full subject data for PCA fitting...")
    
    full_data_list = [[] for _ in range(n_subjects+1)]
    full_labels_list = [[] for _ in range(n_subjects + 1)]
    
    for encoder_inputs, labels, nn_target, masks in tqdm(data_loader, desc="Extracting all data"):
        masks_np = masks.numpy()[:, :n_subjects+1]
        
        for subject_idx in range(n_subjects+1):
            subject_data = encoder_inputs[subject_idx].numpy()
            subject_mask = masks_np[:, subject_idx]
            subj_labels = labels.numpy()[subject_mask]
            
            if subject_mask.any():
                valid_data = subject_data[subject_mask]
                full_data_list[subject_idx].append(valid_data)
                full_labels_list[subject_idx].append(subj_labels)
    
    # Concatenate all batches for each subject
    full_data_list = [np.vstack(subject_data) if subject_data else np.array([]) 
                     for subject_data in full_data_list]
    full_labels_list = [np.vstack(s) if s else np.array([]) for s in full_labels_list]
    
    print("Full data / label shapes:")
    for i, (d, l) in enumerate(zip(full_data_list, full_labels_list)):
        print(f"  Subject {i+1}: data {d.shape}, labels {l.shape}")

    
    return full_data_list, full_labels_list



def extract_common_fmri_samples(data_loader, n_subjects=8):
    """
    Order-safe, single-pass extraction of samples that are valid for *all* subjects
    (including NN, last slot). Guarantees identical row order across all returned arrays.

    Returns:
        common_data_list: list of np.arrays, fMRI data per subject (and NN)
        common_labels: np.array of labels for the same rows
    """
    print("\nExtracting common fMRI samples and labels (single-pass, order-safe)...")

    all_batches   = []          # list of mini-batch tuples (encoder_inputs)
    all_labels    = []          # list of mini-batch labels
    common_masks  = []          # bool vector per batch: True -> keep this row

    for encoder_inputs, labels, nn_target, masks in tqdm(data_loader, desc="Scanning"):
        m = masks.numpy()[:, :n_subjects + 1]         # (B, n_subjects+1)
        keep = m.all(axis=1)                          # (B,)
        all_batches.append(encoder_inputs)            # (list of tensors)
        all_labels.append(labels.numpy())             # (B,)
        common_masks.append(keep)                     # (B,)

    common_data_list = [[] for _ in range(n_subjects + 1)]
    common_labels = []

    for encoder_inputs, labels, keep in zip(all_batches, all_labels, common_masks):
        if keep.any():
            idx = np.where(keep)[0]                   # indices of rows valid for all subjects
            for subj in range(n_subjects + 1):
                common_data_list[subj].append(
                    encoder_inputs[subj].numpy()[idx]
                )
            common_labels.append(labels[idx])

    # Concatenate across batches
    common_data_list = [np.vstack(lst) for lst in common_data_list]
    common_labels = np.concatenate(common_labels)

    print(f"\nFinal common data shapes:")
    for subj, arr in enumerate(common_data_list):
        print(f"  Subject {subj+1}: data={arr.shape}")
    print(f"  Labels: {common_labels.shape}")

    return common_data_list, common_labels


def apply_dimensionality_reduction_robust(full_data_list, full_data_list_test, common_data_list, n_pca=200, n_umap=None, args=None):
    """
    Apply PCA fitted on full data to common samples.
    Now optionally returns PCA models for saving.
    """
    from sklearn.decomposition import PCA
    import umap
    
    reduced_common_list = []
    reduced_full_list = []
    reduced_full_list_test = []
    pca_models = []  
    
    print(f"\nApplying PCA (fit on full data) to common samples...")
    
    for i, (full_data, full_data_test, common_data) in enumerate(zip(full_data_list, full_data_list_test, common_data_list)):
        print(f"  Subject {i+1}: Fitting PCA on {full_data.shape[0]} samples...")
        
        # Fit PCA on FULL data
        pca = PCA(n_components=n_pca, random_state=42)
        pca.fit(full_data)
        pca_models.append(pca) 
        
        # Transform common samples
        common_reduced = pca.transform(common_data)
        print(f"    PCA: {common_data.shape} -> {common_reduced.shape}")

        full_data_reduced = pca.transform(full_data)
        full_data_reduced_test = pca.transform(full_data_test)
        print(f"  Subject {i+1}: Using PCA fitted on {full_data.shape[0]} samples..., transforming {full_data_reduced.shape[0]}")

        import os, umap
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['PYTHONHASHSEED']  = '42'
        
        # Apply UMAP if requested
        if n_umap and n_umap > 0:
            umap_reducer = umap.UMAP(n_components=n_umap,
                                    random_state=42,
                                    n_neighbors=15,
                                    n_jobs=1,               # ← critical
                                    min_dist=0.1,
                                    metric='euclidean')
            common_reduced = umap_reducer.fit_transform(common_reduced)
            print(f"    UMAP: {pca.n_components_} -> {n_umap}")
        
        reduced_common_list.append(common_reduced)
        reduced_full_list.append(full_data_reduced)
        reduced_full_list_test.append(full_data_reduced_test)
    
    # ======== SAVE PCA-TRANSFORMED COMMON SAMPLES ========
    print("\n" + "="*70)
    print("SAVING PCA-TRANSFORMED COMMON SAMPLES")
    print("="*70)

    # The original code relied on os.chdir(VAE_DIR) so these relative 'results/'
    # paths landed in vae/results/. The chdir was removed in the refactor, so
    # ensure the (cwd-relative) results/ dir exists to preserve the same writes.
    os.makedirs('results', exist_ok=True)
    # Use only the basename of save_name so an absolute --save_name does not get
    # embedded into the relative results/ path (byte-identical for bare names).
    _base = os.path.basename(args.save_name)

    # Save the PCA-reduced common samples
    np.savez(
        f'results/pca_common_samples_{_base}',  # e.g., pca_common_samples_cca_alignment_results.npz
        pca_common_data=reduced_common_list,  # List of 8 arrays, each (870, n_pca)
        n_common_samples=870,
        n_pca_components=args.n_pca,
        n_subjects=8,
        subject_shapes=[data.shape for data in reduced_common_list],
        original_sample_indices=None  # You can add this if you track them
    )

    print(f"PCA common samples saved to: results/pca_common_samples_{_base}")

    # Optionally save individual subject files for easier access
    for i, subject_data in enumerate(reduced_common_list):
        np.save(
            f'results/pca_common_samples_subject_{i+1:02d}_{_base.replace(".npz", ".npy")}',
            subject_data
        )
        print(f"  Subject {i+1}: {subject_data.shape} -> results/pca_common_samples_subject_{i+1:02d}_{_base.replace('.npz', '.npy')}")

    # If you want to save the PCA models too (for transforming new data)
    print(f"\nSaving PCA models...")
    for i, pca_model in enumerate(pca_models):
        import pickle
        with open(f'results/pca_model_subject_{i+1:02d}_{_base.replace(".npz", ".pkl")}', 'wb') as f:
            pickle.dump(pca_model, f)
        print(f"  Subject {i+1} PCA model saved")
    
    print("\nReduced common data shapes:")
    for i, data in enumerate(reduced_common_list):
        print(f"  Subject {i+1}: {data.shape}")


    # Check correlations across multiple components
    print("\nPCA correlation across first 10 components:")
    for comp in range(min(10, args.n_pca)):
        corrs = []
        for i in range(len(reduced_common_list)):
            for j in range(i+1, len(reduced_common_list)):
                corr = np.corrcoef(reduced_common_list[i][:, comp], 
                                reduced_common_list[j][:, comp])[0, 1]
                corrs.append(corr)
        print(f"PC{comp+1}: mean={np.mean(corrs):.4f}, std={np.std(corrs):.4f}")
        
        
    return reduced_common_list, pca_models, reduced_full_list, reduced_full_list_test


def extract_pairwise_common_samples(data_loader, n_subjects=8, min_subjects=2, max_subjects=7):
    """
    Extract samples seen by AT LEAST min_subjects (but not necessarily all).
    
    Returns samples where 2+ subjects have data, tracking which subjects saw which images.
    This gives you the images that have partial overlap across subjects.
    
    Parameters:
    -----------
    data_loader : DataLoader
    n_subjects : int
        Number of fMRI subjects (default 8, excluding NN)
    min_subjects : int
        Minimum number of subjects that must have seen an image (default 2)
    
    Returns:
    --------
    pairwise_data_dict : dict
        {
            sample_idx: {
                'data': {subj_idx: array, ...},  # Only subjects that saw it
                'labels': array,
                'subject_mask': boolean array (n_subjects,)
            }
        }
    subject_sample_lists : list of lists
        [subj_0_samples, subj_1_samples, ...] where each is list of sample_idx
    """
    print(f"\nExtracting pairwise common samples (≥{min_subjects} subjects per image)...")
    
    all_batches = []
    all_labels = []
    all_masks = []
    
    # First pass: collect all data
    for encoder_inputs, labels, nn_target, masks in tqdm(data_loader, desc="Scanning batches"):
        m = masks.numpy()[:, :n_subjects]  # Only fMRI subjects, exclude NN
        all_batches.append(encoder_inputs)
        all_labels.append(labels.numpy())
        all_masks.append(m)
    
    # Build dictionary mapping each sample to its data and subject availability
    pairwise_data_dict = {}
    sample_idx = 0
    
    for batch_encoder_inputs, batch_labels, batch_masks in zip(all_batches, all_labels, all_masks):
        batch_size = batch_masks.shape[0]
        
        for local_idx in range(batch_size):
            subject_mask = batch_masks[local_idx]  # (n_subjects,)
            n_available = subject_mask.sum()
            
            # Keep only if at least min_subjects have this image
            if n_available >= min_subjects and n_available <= max_subjects:
                sample_data = {}
                
                # Store data for each subject that has it
                for subj in range(n_subjects):
                    if subject_mask[subj]:
                        sample_data[subj] = batch_encoder_inputs[subj].numpy()[local_idx]
                
                pairwise_data_dict[sample_idx] = {
                    'data': sample_data,
                    'labels': batch_labels[local_idx],
                    'subject_mask': subject_mask,
                    'n_subjects': n_available
                }
            
            sample_idx += 1
    
    # Create subject-wise sample lists
    subject_sample_lists = [[] for _ in range(n_subjects)]
    for sample_idx, sample_info in pairwise_data_dict.items():
        for subj in sample_info['data'].keys():
            subject_sample_lists[subj].append(sample_idx)
    
    # Statistics
    n_samples_by_count = {}
    for sample_info in pairwise_data_dict.values():
        count = sample_info['n_subjects']
        n_samples_by_count[count] = n_samples_by_count.get(count, 0) + 1
    
    print(f"\nExtracted {len(pairwise_data_dict)} samples with ≥{min_subjects} subjects")
    print(f"\nDistribution of samples by number of subjects:")
    for n_subj in sorted(n_samples_by_count.keys()):
        print(f"  {n_subj} subjects: {n_samples_by_count[n_subj]} samples")
    
    print(f"\nSamples per subject:")
    for subj, samples in enumerate(subject_sample_lists):
        print(f"  Subject {subj+1}: {len(samples)} samples")
    
    return pairwise_data_dict, subject_sample_lists
    
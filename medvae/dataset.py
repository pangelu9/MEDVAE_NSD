import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np 
from sklearn.model_selection import train_test_split
import os
from load_data import load_activations, load_labels
import gc

# --- MEDVAE: resolve data locations via the central config ------------
import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from ccn_config import NSD_DATA_DIR, FMRI_FILE_TEMPLATE, FMRI_MINDEYE_TEMPLATE



class Dataset_multiencoder(Dataset):
    """Dataset for multi-encoder VAE with fMRI data from multiple subjects and NN activations"""
    

    def __init__(self, inputs, labels, output, nn_encoder_always_on=True, 
                only_nn_encoder=False, only_fmri_encoders=False, masks=None):
        """
        Args:
            inputs: List of input arrays, one per subject + NN activations (if included)
            labels: Labels/metadata for samples
            output: The target output (NN activations)
            nn_encoder_always_on: Whether to always use the NN activations encoder
            only_nn_encoder: If True, only uses the NN activations encoder
            only_fmri_encoders: If True, only uses the fMRI encoders (0-7)
            masks: List of masks indicating which samples have data for each subject
        """
        self.nn_encoder_always_on = nn_encoder_always_on and not only_fmri_encoders
        self.only_nn_encoder = only_nn_encoder
        self.only_fmri_encoders = only_fmri_encoders

        # Ensure all inputs are numpy arrays first (needed for mask creation)
        for i, input_data in enumerate(inputs):
            if not isinstance(input_data, np.ndarray):
                inputs[i] = np.array(input_data)

        if not isinstance(output, np.ndarray):
            output = np.array(output)

        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # Create masks if not provided
        if masks is None:
            self.masks = []
            
            # Determine which input is NN data (always the last one)
            nn_input_idx = len(inputs) - 1
            
            for i, input_data in enumerate(inputs):
                # Determine if this is the NN input or a fMRI input
                is_nn_input = (i == nn_input_idx and not self.only_fmri_encoders)
                
                if self.only_nn_encoder:
                    # In only_nn_encoder mode, we need different mask handling:
                    if is_nn_input:
                        # NN encoder mask - all ones if it's the NN encoder
                        mask = np.ones(len(input_data), dtype=bool)
                        # IMPORTANT: We still need to determine valid fMRI data for loss calculation
                        print(f"NN encoder mask created with {mask.sum()}/{len(mask)} samples")
                    else:
                        # For fMRI inputs, create mask based on actual data availability
                        # This is for decoder loss calculation, not encoder usage
                        mask = ~np.isnan(input_data).any(axis=1)
                        print(f"fMRI subject {i} mask created with {mask.sum()}/{len(mask)} samples")
                elif self.only_fmri_encoders:
                    # In only_fMRI_encoders mode:
                    if is_nn_input:
                        # We don't use the NN encoder, but we still create a mask for it
                        mask = np.zeros(len(input_data), dtype=bool)
                    else:
                        # fMRI encoder masks based on data availability
                        mask = ~np.isnan(input_data).any(axis=1)
                else:
                    # Normal operation (using all encoders):
                    if is_nn_input and self.nn_encoder_always_on:
                        # NN encoder is always active if specified
                        mask = np.ones(len(input_data), dtype=bool)
                    else:
                        # Create mask based on non-NaN values
                        mask = ~np.isnan(input_data).any(axis=1)
                
                self.masks.append(torch.tensor(mask, dtype=torch.bool))
        else:
            self.masks = masks
        
        # Determine valid samples (at least one encoder has data)
        any_valid = torch.zeros(len(labels), dtype=torch.bool)
        for mask in self.masks:
            any_valid |= mask
        
        # Get indices of valid samples
        self.valid_indices = torch.where(any_valid)[0]
        
        print(f"Dataset created with {len(self.valid_indices)} valid samples out of {len(labels)} total")
        if self.only_nn_encoder:
            print("Using only the NN encoder for encoding, but still tracking valid fMRI data for decoding")
        elif self.only_fmri_encoders:
            print("Using only the fMRI encoders")
        
        # Print mask summary
        print("\nMask summary:")
        for i, mask in enumerate(self.masks):
            if i == len(self.masks) - 1 and not self.only_fmri_encoders:
                print(f"NN encoder/input mask: {mask.sum().item()}/{len(mask)} samples")
            else:
                print(f"fMRI subject {i} mask: {mask.sum().item()}/{len(mask)} samples")

            # Check for samples with both NN and fMRI data
            if len(self.masks) > 1:
                for i in range(len(self.masks)):
                    overlap = (self.masks[-1] & self.masks[i]).sum().item()
                    print(f"Samples with both NN data and Subject {i} fMRI data: {overlap}")

        # ── Pre-convert to tensors (zero-copy where possible) ──────────
        # Replace NaN with 0 so tensor indexing is clean (masks handle validity)
        print("Pre-converting data to tensors...")
        self.inputs = []
        for i, inp in enumerate(inputs):
            if inp.dtype != np.float32:
                inp = inp.astype(np.float32)
            nan_mask = np.isnan(inp)
            if nan_mask.any():
                inp = inp.copy()
                inp[nan_mask] = 0.0
            self.inputs.append(torch.from_numpy(inp))

        if output.dtype != np.float32:
            output = output.astype(np.float32)
        self.output = torch.from_numpy(output)
        self.labels = torch.from_numpy(labels.astype(np.int64))

        # Pre-stack masks into a single (n_samples, n_encoders) tensor
        self.masks_stacked = torch.stack(self.masks, dim=1)  # (n_samples, n_encoders)

        # Pre-compute zero tensors for masked inputs
        self._zero_cache = [torch.zeros(inp.shape[1], dtype=torch.float32) for inp in self.inputs]
        print("Pre-conversion complete.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        # Get inputs for each encoder — pure tensor indexing, no numpy
        encoder_inputs = []
        masks_row = self.masks_stacked[actual_idx]
        for i in range(len(self.inputs)):
            if masks_row[i]:
                encoder_inputs.append(self.inputs[i][actual_idx])
            else:
                encoder_inputs.append(self._zero_cache[i])

        return encoder_inputs, self.labels[actual_idx], self.output[actual_idx], masks_row
    


# Efficient memory
def load_data_multiencoder(args, kwargs, gradual_intro_enc=False):
    """
    Memory-efficient version of load_data_multiencoder that avoids data copying
    """
    import gc
    from memory_utils import get_memory_usage, cleanup_memory
    
    print(f"=== MEMORY-EFFICIENT DATA LOADING ===")
    print(f"Initial memory: {get_memory_usage():.2f} GB")
    
    # Add missing attributes needed by load_functions
    if not hasattr(args, 'brain_to_brain'):
        args.brain_to_brain = False
    if not hasattr(args, 'all_subjects'):
        args.all_subjects = True
    if not hasattr(args, 'shuffle_fmri'):
        args.shuffle_fmri = False
    
    subject_ids =  ["01", "02", "03", "04", "05", "06", "07", "08"]
    args.num_subjects = len (subject_ids)
    
    # Load neural network activations (output target and also input for the extra encoder)
    print("Loading NN activations...")
    nn_activations = load_activations(args)
    print(f"NN Activations shape: {nn_activations.shape}")
    print(f"Memory after NN loading: {get_memory_usage():.2f} GB")
    output_dim = nn_activations.shape[1]
    
    # Load labels
    print("Loading labels...")
    labels = load_labels(args, train_loader_non_shuffled=None)
    if labels.shape[1] > nn_activations.shape[0]:
        labels = labels[:, 0:nn_activations.shape[0]]
    print(f"Labels shape: {labels.shape}")
    print(f"Memory after labels loading: {get_memory_usage():.2f} GB")
    
    # Load all subjects' fMRI data (inputs) with memory limit
    print("Loading fMRI data...")
    noise_level = getattr(args, 'noise_level', 0.0)
    max_memory_gb = getattr(args, 'max_memory_gb', 35.0)
    
    if subject_ids:
        brain_data_list, input_dims = load_brain_activations_multiencoder(
            data_dir=NSD_DATA_DIR + os.sep,
            subject_ids=subject_ids,
            noise_level=noise_level,
            args=args,
            max_memory_gb=max_memory_gb, 
            keep_percent=[100] * len(subject_ids)
        )
    else:
        brain_data_list = []
        input_dims = []
    
    print(f"Memory after fMRI loading: {get_memory_usage():.2f} GB")
    print(f"Loaded {len(brain_data_list)} subjects")

    if getattr(args, 'remove_overlaps', False):
        brain_data_list, nn_activations, labels, input_dims = remove_subject_overlaps(
            brain_data_list=brain_data_list,
            nn_activations=nn_activations, 
            labels=labels,
            subject_ids=subject_ids
        )
    elif getattr(args, 'remove_all_overlaps', False):
        brain_data_list, nn_activations, labels, input_dims = remove_subject_overlaps(
        brain_data_list=brain_data_list,
        nn_activations=nn_activations,
        labels=labels,
        subject_ids=subject_ids,
        remove_all_overlaps=True   # Complete removal
    )
    for i, subject_id in enumerate(subject_ids):
        seed = int(subject_id) 
        subject_data = brain_data_list[i]  
        # =====  PER-SUBJECT SPARSITY (keep among VALID rows) =====
        if isinstance(args.keep_percent, (list, tuple, np.ndarray)):
            val = args.keep_percent[i]
        else:
            val = float(args.keep_percent)

        valid_mask = ~np.isnan(subject_data).any(axis=1)   # rows that already have data
        n_valid   = valid_mask.sum()

        if val > 100:                      # absolute-count mode
            n_keep = min(int(round(val)), n_valid)
        elif val < 100:                    # percentage mode
            n_keep = max(0, int(round(n_valid * val / 100.)))
        else:                              # 100 % → keep every valid row
            n_keep = n_valid
            
        if n_keep == 0:               # wipe everything
            subject_data[:] = np.nan
            print(f"    Sparsity: masked ALL {n_valid} valid images for subject {subject_id}")
        elif n_keep < n_valid:               # decide which VALID rows to discard
            rng = np.random.default_rng(seed)
            discard_idx = rng.choice(np.where(valid_mask)[0],
                                        size=n_valid - n_keep,
                                        replace=False)
            subject_data = subject_data.copy()
            subject_data[discard_idx] = np.nan
            brain_data_list[i] = subject_data 
            print(f"    Sparsity: masked {len(discard_idx)}/{n_valid} valid images "
                    f"(kept {n_keep}) for subject {subject_id}")    
            
    
    # Check encoder usage mode
    only_nn_encoder = getattr(args, 'only_nn_encoder', False)
    only_fmri_encoders = getattr(args, 'only_fmri_encoders', False)
    
    # Handle mutually exclusive options
    if only_nn_encoder and only_fmri_encoders:
        print("Warning: Both only_nn_encoder and only_fmri_encoders are set to True.")
        print("Defaulting to using all encoders.")
        only_nn_encoder = False
        only_fmri_encoders = False
    
    # Prepare final input dimensions
    if only_fmri_encoders:
        print("Using only fMRI encoders")
        input_dims_final = input_dims
    else:
        print("Adding NN activations as additional input")
        input_dims_final = input_dims + [output_dim]
    
    # === EXCLUDE MASK (e.g. OOD category holdout) ===
    exclude_mask_path = getattr(args, 'exclude_mask', None)
    if exclude_mask_path is not None:
        print(f"\n=== APPLYING EXCLUSION MASK ===")
        exclude_mask = np.load(exclude_mask_path)
        n_current = len(nn_activations)
        assert len(exclude_mask) == n_current, \
            f"Exclusion mask length ({len(exclude_mask)}) != data length ({n_current})"
        keep_indices = np.where(~exclude_mask)[0]
        n_excluded = int(exclude_mask.sum())
        n_kept = len(keep_indices)
        print(f"  Loaded mask from {exclude_mask_path}")
        print(f"  Excluding {n_excluded} images, keeping {n_kept}")

        for i in range(len(brain_data_list)):
            brain_data_list[i] = brain_data_list[i][keep_indices]
        nn_activations = nn_activations[keep_indices]
        if labels is not None:
            labels = labels[keep_indices]

        print(f"  Data filtered: {n_current} -> {n_kept} samples")
        gc.collect()
        print(f"  Memory after exclusion: {get_memory_usage():.2f} GB")

    # === FILTER SAMPLES WITH AT LEAST ONE FMRI RESPONSE ===
    print(f"\n=== FILTERING SAMPLES ===")
    filter_enabled = getattr(args, 'filter_no_fmri', False)  # Default to True
    
    if brain_data_list and filter_enabled:
        # Check which samples have at least one valid (non-NaN) fMRI response
        n_samples = len(nn_activations)
        has_fmri_data = np.zeros(n_samples, dtype=bool)
        
        print("Checking fMRI data availability per subject...")
        for subject_idx, brain_data in enumerate(brain_data_list):
            # Check which samples have non-NaN values for this subject
            valid_samples = ~np.isnan(brain_data).all(axis=1)
            has_fmri_data |= valid_samples
            print(f"  Subject {subject_idx+1}: {valid_samples.sum()}/{n_samples} valid samples")
        
        valid_indices = np.where(has_fmri_data)[0]
        n_valid = len(valid_indices)
        n_removed = n_samples - n_valid
        
        print(f"\n  Total samples with at least one fMRI response: {n_valid}/{n_samples}")
        print(f"  Removed {n_removed} samples with only NN encoder data ({100*n_removed/n_samples:.1f}%)")
        
        # Filter all data to only include valid samples
        if n_removed > 0:
            print("  Filtering data...")
            for i in range(len(brain_data_list)):
                brain_data_list[i] = brain_data_list[i][valid_indices]
            nn_activations = nn_activations[valid_indices]
            if labels is not None:
                labels = labels[valid_indices]
            print(f"   Data filtered to {n_valid} samples")
            gc.collect()
            print(f"  Memory after filtering: {get_memory_usage():.2f} GB")
    elif not brain_data_list:
        print("  No fMRI data loaded, skipping filtering")
    else:
        print("  Filtering disabled (filter_no_fmri=False)")

    # === MEMORY-EFFICIENT TRAIN/TEST SPLIT ===
    print(f"\n=== MEMORY-EFFICIENT TRAIN/TEST SPLIT ===")
    print(f"Memory before split: {get_memory_usage():.2f} GB")
    
    # Get number of samples
    n_samples = len(nn_activations)
    print(f"Total samples: {n_samples}")
    
    #  CRITICAL: Split indices instead of data to avoid copying
    indices = np.arange(n_samples)
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=args.test_size, 
        random_state=42
    )
    
    print(f"Split: {len(train_indices)} train, {len(test_indices)} test samples")
    print(f"Memory after index split: {get_memory_usage():.2f} GB")
    
    #  CRITICAL: Create train/test splits efficiently WITHOUT full data copying
    train_inputs_final = []
    test_inputs_final = []
    
    # Process fMRI data with immediate cleanup
    for i, brain_data in enumerate(brain_data_list):
        print(f"Splitting subject {i+1}/{len(brain_data_list)}...")
        
        # Create train/test splits (only copy the slices, not full arrays)
        train_data = brain_data[train_indices].copy()
        test_data = brain_data[test_indices].copy()
        
        train_inputs_final.append(train_data)
        test_inputs_final.append(test_data)
        
        #  CRITICAL: IMMEDIATELY delete the original data to free memory
        brain_data_list[i] = None  # Clear reference
        del brain_data
        gc.collect()
        
        print(f"  Memory after subject {i+1}: {get_memory_usage():.2f} GB")
    
    #  CRITICAL: Clear the brain_data_list completely
    del brain_data_list
    gc.collect()
    print(f"Memory after clearing brain_data_list: {get_memory_usage():.2f} GB")
    

    #Apply the SAME memory-efficient pattern to NN activations
    print("Splitting NN activations...")
    pre_split_memory = get_memory_usage()
    
    # Create splits (only copy the slices, not full arrays) - SAME PATTERN
    train_nn_activations = nn_activations[train_indices].copy()
    test_nn_activations = nn_activations[test_indices].copy()
    
    # Add to inputs if needed
    if not only_fmri_encoders:
        train_inputs_final.append(train_nn_activations)
        test_inputs_final.append(test_nn_activations)
    
    #  CRITICAL: IMMEDIATELY delete the original NN activations (SAME AS fMRI)
    del nn_activations
    gc.collect()
    
    post_split_memory = get_memory_usage()
    print(f"Memory after NN split: {post_split_memory:.2f} GB")
    print(f"NN split memory change: {post_split_memory - pre_split_memory:+.2f} GB")
    
    
    # Handle labels with cleanup
    print("Splitting labels...")
    if labels is not None:
        train_labels = labels[train_indices].copy()
        test_labels = labels[test_indices].copy()
        del labels  #  DELETE ORIGINAL
        gc.collect()
    else:
        train_labels = np.zeros(len(train_indices), dtype=int)
        test_labels = np.zeros(len(test_indices), dtype=int)
    
    print(f"Memory after labels split: {get_memory_usage():.2f} GB")
    
    #  CRITICAL: Clean up indices
    del indices, train_indices, test_indices
    gc.collect()
    
    # === CREATE DATASETS ===
    print(f"\n=== CREATING DATASETS ===")
    print(f"Memory before dataset creation: {get_memory_usage():.2f} GB")
    
    # Create datasets with appropriate encoder configuration
    print("Creating standard datasets...")
    dataset_train = Dataset_multiencoder(
        train_inputs_final,
        train_labels,
        train_nn_activations,
        nn_encoder_always_on=not only_fmri_encoders,
        only_nn_encoder=only_nn_encoder,
        only_fmri_encoders=only_fmri_encoders
    )
    
    print(f"Memory after train dataset: {get_memory_usage():.2f} GB")
    
    dataset_test = Dataset_multiencoder(
        test_inputs_final,
        test_labels,
        test_nn_activations,
        nn_encoder_always_on=not only_fmri_encoders,
        only_nn_encoder=only_nn_encoder,
        only_fmri_encoders=only_fmri_encoders
    )
    
    print(f"Memory after test dataset: {get_memory_usage():.2f} GB")
    
    # Print dataset information
    print("\n=== DATASET SUMMARY ===")
    print(f"Training set:")
    print(f"Target NN activations shape: {train_nn_activations.shape}")
    for i, input_data in enumerate(train_inputs_final):
        if not only_nn_encoder and i < len(subject_ids):
            input_name = f"Subject {subject_ids[i]}"
        else:
            input_name = "NN Activations"
        print(f"{input_name} input shape: {input_data.shape}")
        if hasattr(dataset_train, 'masks') and i < len(dataset_train.masks):
            print(f"{input_name} available samples: {dataset_train.masks[i].sum()}")
    
    # === CREATE DATA LOADERS ===
    print(f"\n=== CREATING DATA LOADERS ===")
    print(f"Memory before data loaders: {get_memory_usage():.2f} GB")
    
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=multiencoder_collate_fn,
        **kwargs
    )
    
    train_loader_non_shuffled = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=multiencoder_collate_fn,
        **kwargs
    )
    
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=multiencoder_collate_fn,
        **kwargs
    )
    
    # Final aggressive cleanup
    cleanup_memory()
    torch.cuda.empty_cache()
    final_memory = get_memory_usage()
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Final memory usage: {final_memory:.2f} GB")
    
    return train_loader, test_loader, train_loader_non_shuffled, input_dims_final, output_dim



def load_brain_activations_multiencoder(
    data_dir: str,
    subject_prefix: str = "fmri_subject",
    subject_ids: Optional[List[str]] = None,
    noise_level: float = 0.0,
    args = None,
    max_memory_gb: float = 35.0,
    keep_percent: float | list[float] = 100.0
) -> Tuple[List[np.ndarray], List[int]]:
    """
    OPTIMIZED VERSION: Converts float64→float32 for 50% memory savings
    Expected usage: ~5.3GB per subject (instead of 10.6GB)

    Args:
        data_dir: Directory containing the brain data files
        subject_prefix: Prefix for subject files
        subject_ids: List of subject IDs to load
        noise_level: Level of noise to add to the data
        args: Arguments object containing dataset configuration
        max_memory_gb: Maximum memory limit in GB
        remove_overlaps: If True, removes overlapping stimuli between subjects,
                        keeping only the first occurrence (from subject with lowest ID)
    """
    from memory_utils import get_memory_usage
    import gc
    
    print(f" OPTIMIZED fMRI loading - Initial memory: {get_memory_usage():.2f} GB")
    print(f"Memory limit: {max_memory_gb} GB")
    print(f"Expected memory per subject: ~5.3GB (with float32 conversion)")
    
    if subject_ids is None:
        subject_files = [f for f in os.listdir(data_dir) if f.startswith(subject_prefix)]
        subject_ids = [int(''.join(filter(str.isdigit, f))) for f in subject_files]
        subject_ids.sort()
    
    brain_data_list = []
    input_dims = []
    
    for i, subject_id in enumerate(subject_ids):
        seed = int(subject_id) 
        # Check memory before loading each subject
        current_memory = get_memory_usage()
        print(f"\nSubject {subject_id}: Memory before loading: {current_memory:.2f} GB")
        
        if current_memory > max_memory_gb:
            print(f"  Memory limit ({max_memory_gb} GB) exceeded. Stopping at subject {subject_id}")
            print(f"   Loaded {len(brain_data_list)} subjects successfully")
            break
        
        # Construct filename
        if args.dataset == "streams":
            filename = FMRI_FILE_TEMPLATE.format(sid=subject_id)
        elif args.dataset == "mindeye":
            filename = FMRI_MINDEYE_TEMPLATE.format(sid=subject_id)
        else:
            filename = f"{subject_prefix}{subject_id}_aligned_ALGO.npz"
        filepath = os.path.join(data_dir, filename)
        
        try:
            #  MEMORY OPTIMIZED LOADING
            print(f"   Loading {filepath}...")
            
            with np.load(filepath) as npz_file:
                fmri_raw = npz_file['fmri_data']
                if getattr(args, 'fmri_shuffle_voxels', False):
                    print("PERMUTING VOXELS")
                    rng = np.random.default_rng(seed=42)   # fixed seed → reproducible
                    perm = rng.permutation(fmri_raw.shape[1])     # one random column permutation
                    fmri_raw = fmri_raw[:, perm]
                                        
                #  KEY OPTIMIZATION: Convert float64 → float32 (50% memory savings)
                if fmri_raw.dtype == np.float64:
                    print(f"   Converting float64 → float32 (50% memory savings)")
                    subject_data = fmri_raw.astype(np.float32)
                else:
                    print(f"   Already float32")
                    subject_data = fmri_raw.copy()
            
            # npz_file automatically closed, memory released
            gc.collect()
            
            memory_after_load = get_memory_usage()
            actual_increase = memory_after_load - current_memory
            expected_size_gb = (subject_data.nbytes) / (1024**3)
            
            print(f"   Memory after loading: {memory_after_load:.2f} GB (+{actual_increase:.2f} GB)")
            print(f"   Expected size: {expected_size_gb:.2f} GB")
            print(f"   Efficiency: {expected_size_gb/actual_increase:.2f}x (1.0 = perfect)")
            

                
            # Data processing
            if hasattr(args, 'shuffle_fmri') and args.shuffle_fmri:
                print("   Shuffling fMRI data")
                np.random.shuffle(subject_data)
            
            # Reshape if necessary
            if len(subject_data.shape) > 2:
                original_shape = subject_data.shape
                subject_data = subject_data.reshape(subject_data.shape[0], -1)
                print(f"   Reshaped: {original_shape} → {subject_data.shape}")
            
            # Add noise if requested (memory-efficient chunked processing)
            if noise_level > 0:
                print(f"   Adding noise (level={noise_level})")
                data_range = np.nanmax(subject_data) - np.nanmin(subject_data)
                noise_scale = data_range * noise_level
                
                # Process in chunks to avoid memory spikes
                chunk_size = 10000
                for start_idx in range(0, subject_data.shape[0], chunk_size):
                    end_idx = min(start_idx + chunk_size, subject_data.shape[0])
                    chunk = subject_data[start_idx:end_idx]
                    
                    # Create noise for this chunk only
                    noise = np.random.normal(0, noise_scale, size=chunk.shape).astype(np.float32)
                    non_nan_mask = ~np.isnan(chunk)
                    chunk[non_nan_mask] += noise[non_nan_mask]
                    
                    subject_data[start_idx:end_idx] = chunk
                    del noise, non_nan_mask  # Clean up immediately
                
                print(f"   Noise added in chunks")
            
            # Store the data
            input_dims.append(subject_data.shape[1])
            brain_data_list.append(subject_data)
            
            final_memory = get_memory_usage()
            total_increase = final_memory - current_memory
            print(f"   Subject {subject_id}: Stored {subject_data.shape}")
            print(f"     Final memory: {final_memory:.2f} GB (+{total_increase:.2f} GB total)")
            
            # Cleanup after every 2 subjects
            if (i + 1) % 2 == 0:
                gc.collect()
                cleanup_memory = get_memory_usage()
                print(f"   Cleanup after subject {subject_id}: {cleanup_memory:.2f} GB")
                
        except FileNotFoundError:
            print(f"  Warning: Data for subject {subject_id} not found at {filepath}")
            continue
        except Exception as e:
            print(f" Error loading subject {subject_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            gc.collect()
            continue
    
    if not brain_data_list:
        raise ValueError("No brain data could be loaded!")
    
    # Final summary
    final_memory = get_memory_usage()
    total_expected = sum((data.nbytes) / (1024**3) for data in brain_data_list)
    
    print(f"\n LOADING COMPLETE!")
    print(f"   Loaded subjects: {len(brain_data_list)}")
    print(f"   Dimensions: {input_dims}")
    print(f"   Final memory: {final_memory:.2f} GB")
    print(f"   Total data size: {total_expected:.2f} GB")
    print(f"   Memory efficiency: {total_expected/final_memory:.2f}x")
    print(f"   Expected subjects at 40GB limit: {40/5.3:.0f} subjects")

    return brain_data_list, input_dims


# Custom collate function for the DataLoader to handle the list of inputs properly
def multiencoder_collate_fn(batch):
    """
    Custom collate function for DataLoader used with Dataset_multiencoder
    
    Args:
        batch: List of (encoder_inputs, label, output, mask) tuples
    
    Returns:
        Tuple of (batched_encoder_inputs, batched_labels, batched_outputs, batched_masks)
    """
    # Unzip the batch
    encoder_inputs_list, labels, outputs, masks = zip(*batch)
    
    # For each subject, stack their inputs separately
    num_subjects = len(encoder_inputs_list[0])
    batched_encoder_inputs = []
    
    for subject_idx in range(num_subjects):
        # Get all inputs for this subject
        subject_inputs = [sample[subject_idx] for sample in encoder_inputs_list]
        # Stack them into a single tensor
        batched_encoder_inputs.append(torch.stack(subject_inputs))
    
    # Stack labels, outputs, and masks
    batched_labels = torch.stack(labels)
    batched_outputs = torch.stack(outputs)
    batched_masks = torch.stack(masks)
    
    return batched_encoder_inputs, batched_labels, batched_outputs, batched_masks


def remove_subject_overlaps(brain_data_list, nn_activations, labels, subject_ids=None, remove_all_overlaps=False):
    """
    Ultra memory-efficient version for when you're really tight on memory.
    """
    print(f"\n REMOVING OVERLAPS (ULTRA-MEMORY-EFFICIENT)...")
    
    if not remove_all_overlaps:
        # Use the simple in-place NaN approach for first-subject priority
        assigned_indices = set()
        
        for i, subject_data in enumerate(brain_data_list):
            print(f"  Subject {subject_ids[i] if subject_ids else i+1}...")
            
            valid_mask = ~np.isnan(subject_data[:, 0])
            valid_indices = np.where(valid_mask)[0]
            
            overlap_count = 0
            for idx in valid_indices:
                if idx in assigned_indices:
                    subject_data[idx] = np.nan
                    overlap_count += 1
                else:
                    assigned_indices.add(idx)
            
            print(f"     {overlap_count} overlaps → NaN")
            
            del valid_mask, valid_indices
            if i % 2 == 0:
                gc.collect()
        
        input_dims = [data.shape[1] for data in brain_data_list]
        return brain_data_list, nn_activations, labels, input_dims
    
    else:
        # For complete removal, fall back to NaN approach to save memory
        print("   Using NaN approach to conserve memory...")
        print("     (Complete row removal would use too much memory)")
        
        # Just remove overlaps by setting to NaN, don't filter global arrays
        # This keeps memory usage minimal
        
        image_counts = {}
        
        # Quick scan to identify overlaps
        for i, subject_data in enumerate(brain_data_list):
            valid_indices = np.where(~np.isnan(subject_data[:, 0]))[0]
            for idx in valid_indices:
                image_counts[idx] = image_counts.get(idx, 0) + 1
        
        overlapping_images = {idx for idx, count in image_counts.items() if count > 1}
        
        # Set overlaps to NaN in all subjects
        total_removed = 0
        for i, subject_data in enumerate(brain_data_list):
            removed = 0
            for idx in overlapping_images:
                if not np.isnan(subject_data[idx, 0]):
                    subject_data[idx] = np.nan
                    removed += 1
            total_removed += removed
            print(f"  Subject {subject_ids[i] if subject_ids else i+1}: {removed} overlaps → NaN")
        
        print(f"   {total_removed} total responses set to NaN")
        print(f"     Arrays maintain original size for memory efficiency")
        
        input_dims = [data.shape[1] for data in brain_data_list]
        return brain_data_list, nn_activations, labels, input_dims
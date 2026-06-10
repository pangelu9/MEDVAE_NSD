"""
Input/Output utilities for saving and loading alignment models (SRM/Procrustes).

This module provides functions to save trained alignment models with all necessary
components (transformation matrices, PCA models, training means) for later use.

Vendored verbatim from ../baselines_old/model_io.py so baselines/ is fully
self-contained for the paper release (loaded locally by the fit_* drivers).
"""

import pickle
import pathlib
from typing import Dict, List, Optional
import numpy as np


def save_alignment_model(
    W_list: List[np.ndarray],
    pca_models: List,
    training_means: List[np.ndarray],
    method_name: str,
    model_dir: str = "alignment_models",
    additional_info: Optional[Dict] = None
) -> pathlib.Path:
    """
    Save trained alignment model (SRM/Procrustes) with all necessary components.
    
    Parameters:
    -----------
    W_list : list of (n_pca_features, n_components) arrays
        Transformation matrices for each subject
    pca_models : list of sklearn PCA objects
        PCA models for transforming to/from PCA space
    training_means : list of (n_pca_features,) arrays
        Mean of training data in PCA space (for centering at inference)
    method_name : str
        Identifier for this model (e.g., "srm_shared872", "procrustes_shared872")
    model_dir : str
        Directory to save model (created if doesn't exist)
    additional_info : dict, optional
        Any additional metadata to store with the model
        
    Returns:
    --------
    filename : pathlib.Path
        Path to saved model file
        
    Examples:
    ---------
    >>> # After training SRM on 872 shared images
    >>> save_alignment_model(
    ...     W_list=W,
    ...     pca_models=pca_models,
    ...     training_means=training_means,
    ...     method_name="srm_shared872",
    ...     additional_info={'n_iter': 20, 'convergence': 'yes'}
    ... )
    """
    model_path = pathlib.Path(model_dir)
    model_path.mkdir(exist_ok=True, parents=True)
    
    # Prepare model dictionary
    model_dict = {
        'W_list': W_list,
        'pca_models': pca_models,
        'training_means': training_means,
        'method_name': method_name,
        'n_subjects': len(W_list),
        'n_components': W_list[0].shape[1] if W_list else 0,
        'n_pca_features': W_list[0].shape[0] if W_list else 0
    }
    
    # Add any additional info
    if additional_info:
        model_dict['additional_info'] = additional_info
    
    # Save to file
    filename = model_path / f"{method_name}_model.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model_dict, f)
    
    print(f"\n{'='*70}")
    print(f"MODEL SAVED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f" File: {filename}")
    print(f"   Method: {method_name}")
    print(f"   Subjects: {model_dict['n_subjects']}")
    print(f"   Components: {model_dict['n_components']}")
    print(f"   PCA features: {model_dict['n_pca_features']}")
    
    if additional_info:
        print(f"   Additional info: {additional_info}")
    
    print(f"{'='*70}\n")
    
    return filename

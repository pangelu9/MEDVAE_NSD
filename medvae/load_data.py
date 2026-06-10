from __future__ import print_function
import numpy as np
import os 
from pathlib import Path

# --- MEDVAE: resolve all data locations via the central config --------
import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from ccn_config import NSD_DIR, NSD_DATA_DIR, NSD_CLASSIFICATION_DIR



# Function to load fMRI data
def load_fmri_data(file_paths, args=None):
    """
    Load fMRI data from specified file paths.
    
    Parameters:
    -----------
    file_paths : list of str
        Paths to fMRI data files for each subject
        
    Returns:
    --------
    data_list : list of numpy arrays
        List containing fMRI data for each subject (samples × voxels)
    """
    print(file_paths)
    data_list = []
    for file_path in file_paths:
        # Check file extension
        if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
            img = nib.load(file_path)
            data = img.get_fdata()
            # Reshape to samples × voxels if needed
            n_samples = data.shape[3] if len(data.shape) > 3 else 1
            data = data.reshape(-1, n_samples).T  # samples × voxels
        elif file_path.endswith('.npz'):
            # Load npz file - you'll need to know which array to extract
            npz_file = np.load(file_path)
            # Assuming there's a main array in the npz file, you might need to adjust this
            # Commonly used keys are 'arr_0' or a custom key name
            if 'fmri_data' in npz_file.keys():
                data = npz_file['fmri_data']
            elif 'responses' in npz_file.keys():
                data = npz_file['responses']
            else:
                # If the key is unknown, use the first array
                data = npz_file[list(npz_file.keys())[0]]
            print(f"Loaded NPZ file with keys: {list(npz_file.keys())}")
        else:
            # For other formats like .npy
            data = np.load(file_path)
        
        if args is not None:
            if args.fmri_shuffle_voxels:
                rng = np.random.default_rng(seed=42)   # fixed seed → reproducible
                perm = rng.permutation(data.shape[1])     # one random column permutation

                data = data[:, perm]
                    
        data_list.append(data)
    
    return data_list


def load_activations(args, train_loader_non_shuffled=False, filename=None):

    if filename == None:
        if args.framework=="multidecoder":
            filename = "activations_all.npy"
        elif args.all_subjects:
            filename = args.filename

    if train_loader_non_shuffled:
        print("train_loader_non_shufflsed")
        filename = "activations_original_aligned.npy"

    #filename = "features_all.npy"

    # If absolute path, use directly; otherwise prepend default data dir
    if os.path.isabs(filename):
        dir = filename
    else:
        dir = os.path.join(NSD_DATA_DIR, filename)
    print("loading data from", dir)

    #data_act = np.load(dir + "activations_final_layers_{}_originall.npy".format(model))  # model 1
    data_act = np.load(dir)#, allow_pickle=True)  # (1000,)
    
    # 
    print("ACTIVATIONS shape", data_act.shape) 
    # data_act = data_act[:,-7168:]
    #activations = stack_dict(dict(data_act))
    activations = data_act# torch.reshape(activations, (activations.shape[0], -1)).numpy()
    print("ACTIVATIONS shape", activations.shape)    
    #activations = (activations - activations.min()) / (activations.max() - activations.min())
    activations = (activations - activations.mean(axis=0)) / (activations.std(axis=0) + 1e-8)
    has_nan = np.isnan(activations).any()
    
    print("after", activations.shape)
    #activations = activations[:, -512:]
    
    has_nan = np.isnan(activations).any()
    print("Does the activations have NaNs?", has_nan)
    

    activations = activations.astype(np.float32)
    return activations


def load_and_concat_labels():
    """
    Load and concatenate labels from all subjects in the same order as load_and_concat_fmri_files().
    Returns the concatenated labels array.
    """
    labels_arrays = []
    
    # Load all label arrays
    for i in range(1, 9):
        subj = f"{i:02d}"
        file_path = Path(os.path.join(NSD_CLASSIFICATION_DIR, f"inference_results_subject_{subj}_all/labels_resnet50.npy"))
        
        try:
            labels = np.load(file_path)
            print(f"Subject {subj} labels shape: {labels.shape}")
            labels_arrays.append(labels)
        except Exception as e:
            print(f"Error loading labels for subject {subj}: {e}")
    
    if not labels_arrays:
        raise ValueError("No label arrays were successfully loaded")
    
    # Concatenate all label arrays
    concatenated_labels = np.concatenate(labels_arrays, axis=0)
    
    #concatenated_labels = concatenated_labels[0:9841,:]
    print(f"\nFinal concatenated labels shape: {concatenated_labels.shape}")
    
    #Delete placeholder column
    label_counts = np.sum(concatenated_labels, axis=0)  # Count occurrences of each label
    min_label_index = np.argmin(label_counts)  # Find the index with the fewest occurrences


    # Remove the column corresponding to the least frequent label
    filtered_data_labels = np.delete(concatenated_labels, min_label_index, axis=1)
    print(f"\nFinal concatenated labels shape: {filtered_data_labels.shape}")

    #np.random.shuffle(filtered_data_labels)

    
    return filtered_data_labels



def load_labels(args, train_loader_non_shuffled=False, filename=None):

    if filename is None:
        if args.brain_to_brain and args.all_subjects and not train_loader_non_shuffled:
            print("brain to brain ALL SUBJECTS")
            ## TODO correct how we load the data, has become very convoluted. :) 
            labels = load_and_concat_labels()
            return labels
        
        elif args.framework=="multidecoder":
            filename = "labels_all.npy"  
        elif args.all_subjects:
            if args.dataset == "mindeye":
                filename = "data/final_datasets_mindeye2/averaged_CLIP/labels_all.npy"
            else:
                filename = "labels_all_aligned.npy"  #labels_all_overl_advers.npy labels_all_overl.npy labels_all_convnext 
            # filename = "labels_resnet50_fair_noise_gaussian3.npy" 
            # filename = "labels_all_aligned_categories.npy"
            
        else:
            filename = "labels_original_aligned.npy" #"labels_original_aligned.npy"
        
        if train_loader_non_shuffled:
            print("train_loader_non_shuffled")      
            filename = "labels_original_aligned.npy"

    dir = os.path.join(NSD_DIR, filename)
    print("loading data from", dir)

    labels = np.load(dir)  # model 1 
    print("labels shape", labels.shape)

    return labels


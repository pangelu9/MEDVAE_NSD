"""Per-encoder VAE loss for the hybrid multi-encoder model."""
import torch
import torch.nn.functional as F


def compute_separate_encoder_losses(model, nn_output, nn_target, fmri_outputs, fmri_inputs,
                                   mu, logvar, masks, kl_beta, args, disabled_encoders=None):
    """
    Compute losses for each encoder separately, properly normalized by valid sample counts
    """
    if disabled_encoders is None:
        disabled_encoders = []
    device = masks.device
    batch_size = masks.size(0)
    num_encoders = mu.size(2)

    # print("MASKS SIZEEEE", masks.shape)
    
    # Initialize loss accumulators
    total_kl_loss = torch.tensor(0.0, device=device)
    total_nn_loss = torch.tensor(0.0, device=device)
    total_fmri_loss = torch.tensor(0.0, device=device)
    
    encoder_kl_losses = {}
    encoder_nn_losses = {}
    encoder_fmri_losses = {}
    encoder_valid_counts = {} # Track valid counts for normalization
    
    # Process each encoder
    for encoder_idx in range(num_encoders):

        if encoder_idx in disabled_encoders:
            # Skip this encoder completely
            encoder_kl_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
            encoder_nn_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
            encoder_fmri_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
            continue

        # Get mask for this encoder
        if args.only_nn_encoder:
            # Get the mask for the NN encoder (last column in masks)
            encoder_mask = masks[:, -1]
        else:
            # Normal behavior - use the matching mask for this encoder
            encoder_mask = masks[:, encoder_idx]

        valid_indices = torch.where(encoder_mask)[0]
        encoder_valid_count = len(valid_indices)

        # print("encoder_valid_count", encoder_valid_count)
        
        encoder_valid_counts[f"encoder_{encoder_idx}"] = {
            "encoder": encoder_valid_count,
            "nn": 0,
            "fmri": {}
        }
        
        if encoder_valid_count == 0:
            encoder_kl_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
            encoder_nn_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
            encoder_fmri_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
            continue
        
        # --- KL LOSS ---
        encoder_mu = mu[valid_indices, :, encoder_idx]
        encoder_logvar = logvar[valid_indices, :, encoder_idx]
        
        logvar_clipped = torch.clamp(encoder_logvar, min=-20, max=20)
        kl_terms = 1 + logvar_clipped - encoder_mu.pow(2) - logvar_clipped.exp()
        # kl_loss = -0.5 * torch.sum(kl_terms, dim=1).mean()  # normalise by valid samples
        
        kl_loss = -0.5 * torch.sum(kl_terms) / batch_size

        # Store KL loss
        encoder_kl_losses[f"encoder_{encoder_idx}"] = kl_loss
        total_kl_loss += kl_loss
        
        # --- NN LOSS ---
        if model.use_nn_decoder and nn_output is not None:

            # Check if NN encoder is frozen
            nn_encoder_frozen = (encoder_idx == model.nn_encoder_idx and 
                                model.nn_encoder_idx >= 0 and
                                all(not p.requires_grad for p in model.encoders[model.nn_encoder_idx].parameters()))
            # Check if NN decoder is frozen
            nn_decoder_frozen = (model.nn_decoder is not None and 
                                all(not p.requires_grad for p in model.nn_decoder.parameters()))
            
            if nn_encoder_frozen and nn_decoder_frozen:
                print(f"Skipping NN→NN loss for encoder {encoder_idx} (both frozen)")
                encoder_nn_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
            else:
                encoder_nn_output = nn_output[valid_indices, :, encoder_idx]
                encoder_nn_target = nn_target[valid_indices]
                # Track valid count for NN loss
                encoder_valid_counts[f"encoder_{encoder_idx}"]["nn"] = encoder_valid_count
                
                nn_loss = F.mse_loss(encoder_nn_output, encoder_nn_target, reduction='sum') / batch_size
                
                encoder_nn_losses[f"encoder_{encoder_idx}"] = nn_loss
                total_nn_loss += nn_loss
        else:
            encoder_nn_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
        
        # --- FMRI LOSS ---
        if model.use_fmri_decoders and fmri_outputs is not None:
            encoder_fmri_loss = torch.tensor(0.0, device=device)
            
            for subject_idx, subject_output in enumerate(fmri_outputs):
                subject_mask = masks[:, subject_idx]
                
                # Apply nn_to_fmri masking if enabled
                if args.nn_to_fmri:
                    # For nn_to_fmri, we need to find samples that have both this encoder and subject data
                    common_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    common_mask[valid_indices] = True
                    effective_mask = subject_mask & common_mask
                else:
                    effective_mask = subject_mask & encoder_mask
                
                common_indices = torch.where(effective_mask)[0]
                valid_count_this_subject = len(common_indices)
                # uncomment for debugging:
                # print("valid_count_this_subject", subject_idx, valid_count_this_subject)
                # print("encoder_idx, subject_idx:", encoder_idx, subject_idx)

                # Track valid count for this subject's fMRI loss
                encoder_valid_counts[f"encoder_{encoder_idx}"]["fmri"][f"subject_{subject_idx}"] = valid_count_this_subject
                
                if valid_count_this_subject > 0:
                    subject_encoder_output = subject_output[common_indices, :, encoder_idx]
                    subject_input = fmri_inputs[subject_idx][common_indices]
                    
                    subject_loss = F.mse_loss(subject_encoder_output, subject_input, reduction='sum') / batch_size # / len(common_indices) # normalise by valid samples
                    # subject_loss = F.mse_loss(subject_encoder_output, subject_input, reduction='mean')
                    encoder_fmri_loss += subject_loss
            
            encoder_fmri_losses[f"encoder_{encoder_idx}"] = encoder_fmri_loss
            total_fmri_loss += encoder_fmri_loss
        else:
            encoder_fmri_losses[f"encoder_{encoder_idx}"] = torch.tensor(0.0, device=device)
    
    # Augment return dictionary with valid counts for normalization in training/testing loops
    return {
        'total_kl': total_kl_loss,
        'total_nn': total_nn_loss,
        'total_fmri': total_fmri_loss,
        'encoder_kl': encoder_kl_losses,
        'encoder_nn': encoder_nn_losses,
        'encoder_fmri': encoder_fmri_losses,
        'valid_counts': encoder_valid_counts
    }

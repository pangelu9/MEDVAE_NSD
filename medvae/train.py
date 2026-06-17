import numpy as np
import torch
import gc
from loss import compute_separate_encoder_losses


def get_cyclical_beta(batch_idx, batches_per_epoch, cycle_length):
    """
    Calculate cyclical beta value for KL annealing
    
    Args:
        batch_idx: Current batch index
        batches_per_epoch: Number of batches per epoch
        cycle_length: Length of one cycle in batches
    
    Returns:
        Beta factor (between 0 and 1)
    """
    # Calculate current position in the cycle
    cycle_idx = (batch_idx % cycle_length) / cycle_length
    
    # Use sinusoidal function for smooth transition
    beta_factor = 0.5 * (1 - np.cos(np.pi * cycle_idx))
    
    return beta_factor


def train_hybrid_vae(model, train_loader, device, optimizer, epoch, args, total_annealing_steps=0, disabled_encoders=None, no_grad_update=False):
    
    if disabled_encoders is None:
        disabled_encoders = []

    if no_grad_update:
        model.eval()
    else:
        model.train()
    
    # Initialize metrics
    train_loss = 0
    nn_rec_loss = 0
    nn_rec_loss_fmri_pathway = 0  # fMRI encoder → NN decoder
    nn_rec_loss_ann_pathway = 0   # ANN encoder → NN decoder
    fmri_rec_loss = 0
    fmri_rec_loss_fmri_pathway = 0  # fMRI encoder → fMRI decoder
    fmri_rec_loss_ann_pathway = 0   # ANN encoder → fMRI decoder
    KLD_loss = 0
    align_loss = 0
    total_samples = 0
    num_batches_processed = 0
    # Initialize counters for proper normalization (similar to test function)
    nn_valid_count = 0
    fmri_valid_count = 0
    kl_valid_count = 0
    
    min_batch_size = getattr(args, 'min_batch_size', 1)
    
    # Set KL annealing
    kl_beta = args.KL_beta
    nn_weight = args.nn_weight
    ann_pathway_weight = args.ann_pathway_weight if args.ann_pathway_weight is not None else args.nn_weight
    fmri_weight = args.fmri_weight
    fmri_pathway_weight = args.fmri_pathway_weight if args.fmri_pathway_weight is not None else args.fmri_weight

    for batch_idx, (encoder_inputs, labels, nn_target, masks) in enumerate(train_loader):
        batch_size = nn_target.size(0)
        
        # Skip very small batches
        if batch_size < min_batch_size:
            print(f"Skipping batch {batch_idx} with size {batch_size} < minimum {min_batch_size}")
            continue
            
        total_samples += batch_size
        
        # Update beta for KL annealing
        if args.kl_annealing == "cyclical":
            cycle_length = len(train_loader) // 4
            beta_factor = get_cyclical_beta(batch_idx, len(train_loader), cycle_length)
            kl_beta = args.KL_beta * beta_factor
        elif args.kl_annealing == "monotonic":
            kl_beta = args.KL_beta * min(1, (((epoch-1)*len(train_loader)+batch_idx)/total_annealing_steps))

        # Set weight annealing for nn_weight and fmri_weight

        if hasattr(args, 'weight_annealing') and args.weight_annealing is not None:
            weight_annealing_epochs = getattr(args, 'weight_annealing_epochs', args.epochs)
            total_weight_annealing_steps = weight_annealing_epochs * len(train_loader)
            current_step = (epoch - 1) * len(train_loader) + batch_idx
            
            if args.weight_annealing == "fmri":
                # Increasing: go from 0 to target over weight_annealing_epochs
                if current_step <= total_weight_annealing_steps:
                    progress = current_step / total_weight_annealing_steps
                    fmri_weight = min(args.fmri_weight * progress, 1)
                    fmri_pathway_target = args.fmri_pathway_weight if args.fmri_pathway_weight is not None else args.fmri_weight
                    fmri_pathway_weight = min(fmri_pathway_target * progress, 1)
                else:
                    fmri_weight = args.fmri_weight
                    fmri_pathway_weight = args.fmri_pathway_weight if args.fmri_pathway_weight is not None else args.fmri_weight
                    
            elif args.weight_annealing == "nn":
                # Decreasing: go from args.nn_weight to 1 over weight_annealing_epochs
                if current_step <= total_weight_annealing_steps:
                    progress = current_step / total_weight_annealing_steps
                    nn_weight = max(args.nn_weight * (1 - progress) + 1 * progress, 1)
                else:
                    nn_weight = 1
        
        # Move data to device
        encoder_inputs = [x.to(device) for x in encoder_inputs]
        nn_target = nn_target.to(device)
        all_masks = masks.clone()
        all_masks = all_masks.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        nn_output, fmri_outputs, mu, logvar = model(encoder_inputs, all_masks)

        # Alignment / RSA loss terms were exploratory (weight 0 in the paper);
        # the weighted terms are removed so align_loss logging stays 0.
        align_loss_val = 0.0
        rsa_loss_val = 0.0

        # Compute loss using the separate encoder losses function
        encoder_losses = compute_separate_encoder_losses(
            model, nn_output, nn_target, fmri_outputs, encoder_inputs,
            mu, logvar, all_masks, kl_beta, args
        )


        # Get total losses
        nn_loss_fmri_pathway = 0  # fMRI encoder → NN decoder
        nn_loss_ann_pathway = 0   # ANN encoder → NN decoder
        fmri_loss = 0
        fmri_loss_fmri_pathway = 0  # fMRI encoder → fMRI decoder
        fmri_loss_ann_pathway = 0   # ANN encoder → fMRI decoder
        kl_loss = 0

        # Track valid counts for this batch (similar to test function)
        batch_kl_count = 0
        batch_nn_count = 0
        batch_fmri_count = 0

        for i in range(mu.size(2)):
            if i in disabled_encoders:
                continue  # Skip disabled encoders

            if i == model.nn_encoder_idx:
                nn_loss_ann_pathway += encoder_losses['encoder_nn'][f'encoder_{i}']
                fmri_loss_ann_pathway += encoder_losses['encoder_fmri'][f'encoder_{i}']
            else:
                nn_loss_fmri_pathway += encoder_losses['encoder_nn'][f'encoder_{i}']
                fmri_loss_fmri_pathway += encoder_losses['encoder_fmri'][f'encoder_{i}']
            fmri_loss += encoder_losses['encoder_fmri'][f'encoder_{i}']
            kl_loss += encoder_losses['encoder_kl'][f'encoder_{i}']

            # Get valid counts for proper normalization
            valid_counts = encoder_losses['valid_counts'][f'encoder_{i}']
            batch_kl_count += valid_counts['encoder']
            batch_nn_count += valid_counts['nn']
            batch_fmri_count += sum(count for _, count in valid_counts['fmri'].items())

        # Update total valid counts
        kl_valid_count += batch_kl_count
        nn_valid_count += batch_nn_count
        fmri_valid_count += batch_fmri_count

        nn_loss = nn_loss_fmri_pathway + nn_loss_ann_pathway  # total for logging

        # Calculate total loss for backpropagation (un-normalized)
        loss = kl_beta * kl_loss + nn_weight * nn_loss_fmri_pathway + ann_pathway_weight * nn_loss_ann_pathway + fmri_pathway_weight * fmri_loss_fmri_pathway + fmri_weight * fmri_loss_ann_pathway
        
        # Backward pass and optimization
        if no_grad_update:
            print("train function with no grad update")
        else:
            loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update accumulators (un-normalized)
        train_loss += loss.item()
        nn_rec_loss += nn_loss.item() if isinstance(nn_loss, torch.Tensor) else nn_loss
        nn_rec_loss_fmri_pathway += nn_loss_fmri_pathway.item() if isinstance(nn_loss_fmri_pathway, torch.Tensor) else nn_loss_fmri_pathway
        nn_rec_loss_ann_pathway += nn_loss_ann_pathway.item() if isinstance(nn_loss_ann_pathway, torch.Tensor) else nn_loss_ann_pathway
        fmri_rec_loss += fmri_loss.item()
        fmri_rec_loss_fmri_pathway += fmri_loss_fmri_pathway.item() if isinstance(fmri_loss_fmri_pathway, torch.Tensor) else fmri_loss_fmri_pathway
        fmri_rec_loss_ann_pathway += fmri_loss_ann_pathway.item() if isinstance(fmri_loss_ann_pathway, torch.Tensor) else fmri_loss_ann_pathway
        KLD_loss += kl_loss.item()
        num_batches_processed += 1
        align_loss += align_loss_val.item() if isinstance(align_loss_val, torch.Tensor) else align_loss_val
        
        # Step optimizer
        optimizer.step()
        
        if batch_idx % 10 == 0:  # Every 10 batches
            torch.cuda.empty_cache()
            gc.collect()

        # Log progress with per-sample normalization for display only
        if batch_idx % args.log_interval == 0:
            # Display batch statistics with per-sample normalization for readability
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (NN: {:.6f}, fMRI: {:.6f}, KL: {:.6f}, Align: {:.6f})'.format(
                epoch, total_samples, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
                nn_loss.item(),
                fmri_loss.item(),
                kl_loss.item(),
                align_loss_val.item() if isinstance(align_loss_val, torch.Tensor) else 0.0))
            
            # Optionally print per-encoder metrics with proper normalization
            for i in range(mu.size(2)):
                if i in disabled_encoders:
                    continue 

                encoder_kl = encoder_losses['encoder_kl'][f'encoder_{i}'].item()
                encoder_nn = encoder_losses['encoder_nn'][f'encoder_{i}'].item()
                encoder_fmri = encoder_losses['encoder_fmri'][f'encoder_{i}'].item()
                
                # Get valid counts
                valid_counts = encoder_losses['valid_counts'][f'encoder_{i}']
                encoder_count = valid_counts['encoder']
                nn_count = valid_counts['nn']
                total_fmri_count = sum(count for _, count in valid_counts['fmri'].items())
                
                # Uncomment if detailed per-encoder stats are needed
                #     
                #     print(f"  Encoder {i} (encoder={encoder_count}, nn={nn_count}, fmri={total_fmri_count}): "
                #         f"KL={kl_normalized:.4f}, NN={nn_normalized:.4f}, fMRI={fmri_normalized:.4f}")
            
            del encoder_inputs, labels, nn_target, masks
            del nn_output, fmri_outputs, mu, logvar  # Model outputs
            torch.cuda.empty_cache()
    
    # Handle the case where all batches were skipped
    if total_samples == 0:
        print("WARNING: No samples were processed in this epoch")
        return 0.0, 0.0, 0.0, 0.0, 0.0, kl_beta
    
    # Perform proper normalization at the end using valid counts
    KLD_loss_normalized = KLD_loss  / num_batches_processed
    nn_rec_loss_normalized = nn_rec_loss / num_batches_processed
    nn_rec_loss_fmri_pathway_norm = nn_rec_loss_fmri_pathway / num_batches_processed
    nn_rec_loss_ann_pathway_norm = nn_rec_loss_ann_pathway / num_batches_processed
    fmri_rec_loss_normalized = fmri_rec_loss / num_batches_processed
    fmri_rec_loss_fmri_pathway_norm = fmri_rec_loss_fmri_pathway / num_batches_processed
    fmri_rec_loss_ann_pathway_norm = fmri_rec_loss_ann_pathway / num_batches_processed

    # Recompute total loss with normalized components
    train_loss_normalized = kl_beta * KLD_loss_normalized + nn_weight * nn_rec_loss_fmri_pathway_norm + ann_pathway_weight * nn_rec_loss_ann_pathway_norm + fmri_pathway_weight * fmri_rec_loss_fmri_pathway_norm + fmri_weight * fmri_rec_loss_ann_pathway_norm
    
    # Normalize alignment loss by total samples (it affects all samples)
    align_loss_normalized = align_loss / total_samples
    
    # Print epoch summary with properly normalized metrics
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss_normalized))
    print('====> Average NN loss: {:.4f} ({} valid samples)'.format(
          nn_rec_loss_normalized, nn_valid_count))
    print('====> Average fMRI loss: {:.4f} ({} valid samples)'.format(
          fmri_rec_loss_normalized, fmri_valid_count))
    print('====> Average KL loss: {:.4f} ({} valid samples)'.format(
          KLD_loss_normalized, kl_valid_count))
    
    # Return properly normalized metrics
    return (
        train_loss_normalized,
        nn_rec_loss_normalized,
        fmri_rec_loss_normalized,
        KLD_loss_normalized,
        align_loss_normalized,
        kl_beta
    )


def test_hybrid_vae(model, test_loader, device, epoch, args, kl_beta, disabled_encoders=None):
    """
    Test function for hybrid VAE with separate encoder latent spaces and proper normalization
    """
    if disabled_encoders is None:
        disabled_encoders = []

    model.eval()
    
    # Initialize metrics
    test_loss = 0
    nn_rec_loss = 0
    fmri_rec_loss = 0
    KLD_loss = 0
    num_batches_processed = 0

    
    # Initialize metrics for all encoders (including NN)
    test_loss_all = 0
    nn_rec_loss_all = 0
    nn_rec_loss_all_fmri_pathway = 0
    nn_rec_loss_all_ann_pathway = 0
    fmri_rec_loss_all = 0
    fmri_rec_loss_all_fmri_pathway = 0
    fmri_rec_loss_all_ann_pathway = 0
    KLD_loss_all = 0

    # Initialize metrics for NN encoders 
    test_loss_nn = 0
    nn_rec_loss_nn = 0
    fmri_rec_loss_nn = 0
    KLD_loss_nn = 0

    # Initialize sample counters for proper normalization
    total_samples = 0
    nn_valid_count = 0
    fmri_valid_count = 0
    kl_valid_count = 0
    
    # Similar counters for fMRI-only evaluation
    nn_valid_count_fmri_only = 0
    fmri_valid_count_fmri_only = 0
    kl_valid_count_fmri_only = 0
    
    align_loss = 0
    min_batch_size = getattr(args, 'min_batch_size', 1)
    eval_only_fmri = getattr(args, 'eval_only_fmri', False)
    
    with torch.no_grad():
        for batch_idx, (encoder_inputs, labels, nn_target, masks) in enumerate(test_loader):
            batch_size = nn_target.size(0)
            
            # Skip very small batches
            if batch_size < min_batch_size:
                continue
                
            total_samples += batch_size
            
            # Move data to device
            encoder_inputs = [x.to(device) for x in encoder_inputs]
            nn_target = nn_target.to(device)
            all_masks = masks.clone() 
            all_masks = all_masks.to(device)
            
            try:
                # Calculate alignment loss
                align_loss_val = 0.0
                
                # PART 1: Always calculate metrics with all encoders
                # Forward pass with all encoders
                nn_output_all, fmri_outputs_all, mu_all, logvar_all = model(encoder_inputs, all_masks)

                # Calculate losses for all encoders with proper normalization
                encoder_losses_all = compute_separate_encoder_losses(
                    model, nn_output_all, nn_target, fmri_outputs_all, encoder_inputs,
                    mu_all, logvar_all, all_masks, kl_beta, args
                )
                
                # Get loss values but skip disabled encoders
                kl_loss_all = 0
                nn_loss_all = 0
                nn_loss_all_fmri_pathway = 0
                nn_loss_all_ann_pathway = 0
                fmri_loss_all = 0
                fmri_loss_all_fmri_pathway = 0
                fmri_loss_all_ann_pathway = 0

                for i in range(mu_all.size(2)):
                    if i in disabled_encoders:
                        continue  # Skip disabled encoders

                    kl_loss_all += encoder_losses_all['encoder_kl'][f'encoder_{i}']
                    enc_nn = encoder_losses_all['encoder_nn'][f'encoder_{i}']
                    enc_fmri = encoder_losses_all['encoder_fmri'][f'encoder_{i}']
                    if i == model.nn_encoder_idx:
                        nn_loss_all_ann_pathway += enc_nn
                        fmri_loss_all_ann_pathway += enc_fmri
                    else:
                        nn_loss_all_fmri_pathway += enc_nn
                        fmri_loss_all_fmri_pathway += enc_fmri
                    nn_loss_all += enc_nn
                    fmri_loss_all += enc_fmri
                
                # Get valid counts for each loss component
                all_valid_counts = encoder_losses_all['valid_counts']
                batch_kl_count = sum(counts['encoder'] for idx, counts in all_valid_counts.items() 
                      if int(idx.split('_')[1]) not in disabled_encoders)
                batch_nn_count = sum(counts['nn'] for idx, counts in all_valid_counts.items() 
                                if int(idx.split('_')[1]) not in disabled_encoders)
                batch_fmri_count = sum(sum(subject_count for subject, subject_count in counts['fmri'].items()) 
                                    for idx, counts in all_valid_counts.items() 
                                    if int(idx.split('_')[1]) not in disabled_encoders)
                
                # Update overall valid counts
                kl_valid_count += batch_kl_count
                nn_valid_count += batch_nn_count
                fmri_valid_count += batch_fmri_count
                
                # Total weighted loss for all encoders
                ann_pw = args.ann_pathway_weight if args.ann_pathway_weight is not None else args.nn_weight
                fmri_pw = args.fmri_pathway_weight if args.fmri_pathway_weight is not None else args.fmri_weight
                loss_all = kl_beta * kl_loss_all + args.nn_weight * nn_loss_all_fmri_pathway + ann_pw * nn_loss_all_ann_pathway + fmri_pw * fmri_loss_all_fmri_pathway + args.fmri_weight * fmri_loss_all_ann_pathway

                # Update accumulators for all-encoder metrics
                test_loss_all += loss_all.item()
                nn_rec_loss_all += nn_loss_all.item()
                nn_rec_loss_all_fmri_pathway += nn_loss_all_fmri_pathway.item() if isinstance(nn_loss_all_fmri_pathway, torch.Tensor) else nn_loss_all_fmri_pathway
                nn_rec_loss_all_ann_pathway += nn_loss_all_ann_pathway.item() if isinstance(nn_loss_all_ann_pathway, torch.Tensor) else nn_loss_all_ann_pathway
                fmri_rec_loss_all += fmri_loss_all.item()
                fmri_rec_loss_all_fmri_pathway += fmri_loss_all_fmri_pathway.item() if isinstance(fmri_loss_all_fmri_pathway, torch.Tensor) else fmri_loss_all_fmri_pathway
                fmri_rec_loss_all_ann_pathway += fmri_loss_all_ann_pathway.item() if isinstance(fmri_loss_all_ann_pathway, torch.Tensor) else fmri_loss_all_ann_pathway
                KLD_loss_all += kl_loss_all.item()
                num_batches_processed += 1

                # PART 2: If eval_only_fmri is set, calculate metrics with only fMRI encoders
                if eval_only_fmri:
                    # Create a copy of masks with NN encoder disabled
                    fmri_masks = all_masks.clone()
                    if fmri_masks.shape[1] > len(model.encoders) - 1:
                        fmri_masks[:, -1] = False
                    
                    # Forward pass with only fMRI encoders
                    nn_output_fmri, fmri_outputs_fmri, mu_fmri, logvar_fmri = model(encoder_inputs, fmri_masks)

                    # Calculate losses for fMRI-only with proper normalization
                    encoder_losses_fmri = compute_separate_encoder_losses(
                        model, nn_output_fmri, nn_target, fmri_outputs_fmri, encoder_inputs,
                        mu_fmri, logvar_fmri, fmri_masks, kl_beta, args
                    )
                    
                    # Get loss values
                    kl_loss_fmri = encoder_losses_fmri['total_kl']
                    nn_loss_fmri = encoder_losses_fmri['total_nn']
                    fmri_loss_fmri = encoder_losses_fmri['total_fmri']

                    # NN-only losses = All encoders losses - fMRI-only losses
                    kl_loss_nn_only = kl_loss_all - kl_loss_fmri
                    nn_loss_nn_only = nn_loss_all - nn_loss_fmri
                    fmri_loss_nn_only = fmri_loss_all - fmri_loss_fmri
                    
                    # Get valid counts for each loss component
                    fmri_only_valid_counts = encoder_losses_fmri['valid_counts']
                    batch_kl_count_fmri = sum(counts['encoder'] for encoder, counts in fmri_only_valid_counts.items())
                    batch_nn_count_fmri = sum(counts['nn'] for encoder, counts in fmri_only_valid_counts.items())
                    batch_fmri_count_fmri = sum(sum(subject_count for subject, subject_count in counts['fmri'].items()) 
                                            for encoder, counts in fmri_only_valid_counts.items())
                    
                    # Update overall valid counts for fMRI-only
                    kl_valid_count_fmri_only += batch_kl_count_fmri
                    nn_valid_count_fmri_only += batch_nn_count_fmri
                    fmri_valid_count_fmri_only += batch_fmri_count_fmri
                    
                    # Total weighted loss for fMRI[enc]-only (fmri_loss_fmri is entirely fMRI→fMRI)
                    loss_fmri = kl_beta * kl_loss_fmri + args.nn_weight * nn_loss_fmri + fmri_pw * fmri_loss_fmri
                    # Total weighted loss for NN[enc]-only (fmri_loss_nn_only is entirely ANN→fMRI)
                    loss_fmri_nn_only = kl_beta * kl_loss_nn_only + ann_pw * nn_loss_nn_only + args.fmri_weight * fmri_loss_nn_only

                    # Update accumulators for fMRI-only metrics
                    test_loss += loss_fmri.item()
                    nn_rec_loss += nn_loss_fmri.item()
                    fmri_rec_loss += fmri_loss_fmri.item()
                    KLD_loss += kl_loss_fmri.item()

                    test_loss_nn += loss_fmri_nn_only.item()
                    nn_rec_loss_nn += nn_loss_nn_only.item()
                    fmri_rec_loss_nn += fmri_loss_nn_only.item()
                    KLD_loss_nn += kl_loss_nn_only.item()
                else:
                    # If not evaluating fMRI-only, use the all-encoder metrics
                    test_loss += loss_all.item()
                    nn_rec_loss += nn_loss_all.item()
                    fmri_rec_loss += fmri_loss_all.item()
                    KLD_loss += kl_loss_all.item()
                    
                    # Use the same valid counts
                    kl_valid_count_fmri_only = kl_valid_count
                    nn_valid_count_fmri_only = nn_valid_count
                    fmri_valid_count_fmri_only = fmri_valid_count
                
                align_loss += align_loss_val.item() if isinstance(align_loss_val, torch.Tensor) else align_loss_val
                
            except RuntimeError as e:
                print(f"Error in test batch {batch_idx}: {str(e)}")
                torch.cuda.empty_cache()
                continue
    
    # Handle the case where all batches were skipped
    if total_samples == 0:
        print("WARNING: No samples were processed during testing")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0], 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Normalize losses by actual valid sample counts
    # For all encoders
    KLD_loss_all = KLD_loss_all / num_batches_processed
    nn_rec_loss_all = nn_rec_loss_all / num_batches_processed
    nn_rec_loss_all_fmri_pathway = nn_rec_loss_all_fmri_pathway / num_batches_processed
    nn_rec_loss_all_ann_pathway = nn_rec_loss_all_ann_pathway / num_batches_processed
    fmri_rec_loss_all = fmri_rec_loss_all / num_batches_processed
    fmri_rec_loss_all_fmri_pathway = fmri_rec_loss_all_fmri_pathway / num_batches_processed
    fmri_rec_loss_all_ann_pathway = fmri_rec_loss_all_ann_pathway / num_batches_processed
    ann_pw = args.ann_pathway_weight if args.ann_pathway_weight is not None else args.nn_weight
    fmri_pw = args.fmri_pathway_weight if args.fmri_pathway_weight is not None else args.fmri_weight
    test_loss_all = kl_beta * KLD_loss_all + args.nn_weight * nn_rec_loss_all_fmri_pathway + ann_pw * nn_rec_loss_all_ann_pathway + fmri_pw * fmri_rec_loss_all_fmri_pathway + args.fmri_weight * fmri_rec_loss_all_ann_pathway

    # For fMRI-only or current config (fMRI-only path has no ANN encoder, so nn_weight applies)
    KLD_loss = KLD_loss / num_batches_processed
    nn_rec_loss = nn_rec_loss / num_batches_processed
    fmri_rec_loss = fmri_rec_loss / num_batches_processed
    # fMRI-only path: fmri_rec_loss is entirely fMRI→fMRI, so use fmri_pw
    test_loss = kl_beta * KLD_loss + args.nn_weight * nn_rec_loss + fmri_pw * fmri_rec_loss
    
    align_loss /= total_samples
    
    # Print test summary with proper normalization
    if eval_only_fmri:
        print('====> Test set normalized losses (fMRI encoders only):')
        print(f'  Total: {test_loss:.4f}, NN: {nn_rec_loss:.4f}, fMRI: {fmri_rec_loss:.4f}, KL: {KLD_loss:.4f}')
        print(f'  Valid counts: KL: {kl_valid_count_fmri_only}, NN: {nn_valid_count_fmri_only}, fMRI: {fmri_valid_count_fmri_only}')
        
        print('====> Test set normalized losses (all encoders):')
        print(f'  Total: {test_loss_all:.4f}, NN: {nn_rec_loss_all:.4f}, fMRI: {fmri_rec_loss_all:.4f}, KL: {KLD_loss_all:.4f}')
        print(f'  Valid counts: KL: {kl_valid_count}, NN: {nn_valid_count}, fMRI: {fmri_valid_count}')
    else:
        print('====> Test set normalized losses:')
        print(f'  Total: {test_loss:.4f}, NN: {nn_rec_loss:.4f}, fMRI: {fmri_rec_loss:.4f}, KL: {KLD_loss:.4f}')
        print(f'  Valid counts: KL: {kl_valid_count}, NN: {nn_valid_count}, fMRI: {fmri_valid_count}')
    
    print('====> Test set alignment loss: {:.4f}'.format(align_loss))
    
    # Silhouette / cross-subject alignment metrics are computed post-hoc by the
    # evaluation entry (evaluate_vae.py), not during training.
    fmri_combined_silhouette = 0
    full_combined_silhouette = 0
    avg_encoder_silhouette = 0
    per_encoder_silhouettes = [0] * len(model.encoders)
    nn_encoder_silhouette = 0

    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Return all metrics including both sets of losses
    # Note: We're returning normalized losses here!
    return (
        test_loss, nn_rec_loss, fmri_rec_loss, KLD_loss, align_loss, 
        fmri_combined_silhouette, full_combined_silhouette, avg_encoder_silhouette, 
        per_encoder_silhouettes, nn_encoder_silhouette,
        test_loss_all, nn_rec_loss_all, fmri_rec_loss_all, KLD_loss_all
    )

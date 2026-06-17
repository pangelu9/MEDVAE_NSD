import torch
import torch.optim as optim
import numpy as np

import os

# --- MEDVAE: resolve data locations via the central config ------------
import sys as _sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from ccn_config import RESULTS_DIR

import pandas as pd

from pipeline import train_test_epoch, load_data, create_vae_model
from common_samples import find_common_samples
from args import parse_args

import os


def main():   
    # Parse command-line arguments
    args = parse_args()

    # Set up device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    # Seed every RNG for reproducibility (python / numpy / torch / cuda)
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # MEDVAE: ensure the results/checkpoint output directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ########## FIND AND KEEP ONLY SHARED DATA ##########
    fmri_common, nn_common, labels_common, idx_common = find_common_samples(args)
    args.fmri_common = fmri_common
    args.nn_common = nn_common
    args.labels_common = labels_common

    # Load data

    if args.train:
        train_loader, test_loader, train_loader_non_shuffled, input_dims, output_dim = load_data(args, kwargs, gradual_intro_enc=False)

        # hybrid_vae handles the input_dim differently (passed directly from args)
        if args.hybrid_vae:
            # hybrid_vae handles differently the input_dim, I do not have to provide it like for the other parts of my code e.g., the previous versions that were doing fmri->nn only
            input_dims = args.input_dim
            output_dim=args.output_dim

    else:
        input_dims=args.input_dim
        output_dim=args.output_dim

    model = create_vae_model(args, input_dims, output_dim)
    model.to(device)

    print("\nVAE Architecture Dimensions:")

    if args.load_name:
        print("Loading model", args.load_name)
        checkpoint = torch.load(os.path.join(RESULTS_DIR, args.load_name), weights_only=False)

        # Handle finetuned checkpoints: load base model first, then overlay finetuned weights
        if checkpoint.get('finetune', False):
            base_model_name = checkpoint['base_model']
            print(f"  Finetuned checkpoint detected. Loading base model: {base_model_name}")
            base_checkpoint = torch.load(os.path.join(RESULTS_DIR, base_model_name), weights_only=False)
            model.load_state_dict(base_checkpoint["model_state_dict"])
            # Overlay finetuned weights
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"  Overlaid {len(checkpoint['model_state_dict'])} finetuned parameters")
        else:
            state_dict = checkpoint["model_state_dict"]
            model.load_state_dict(state_dict)

        print("\nFreezing NN encoder and decoder...")

        # Freeze NN encoder
        if model.nn_encoder_idx >= 0:
            for param in model.encoders[model.nn_encoder_idx].parameters():
                param.requires_grad = False
            print(f"   Froze NN encoder at index {model.nn_encoder_idx}")

        # Freeze NN decoder
        if model.nn_decoder is not None:
            for param in model.nn_decoder.parameters():
                param.requires_grad = False
            print(f"   Froze NN decoder")

        
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Calculate total annealing steps for KL divergence
    if args.kl_annealing != 'none':
        total_annealing_steps = args.epochs * len(train_loader)
    else:
        total_annealing_steps = 0

    train_losses = []
    test_losses = []
    train_align_losses = []
    test_align_losses = []
    kl_betas = []
    combined_silhouettes = []
    full_combined_silhouettes = []
    avg_encoder_silhouettes = []  
    all_per_encoder_silhouettes = []
    nn_encoder_silhouettes = []
    test_losses_all = []

    if args.train:
        print("Start training:")
        for epoch in range(1, args.epochs + 1):

            train_losses, test_losses, train_align_losses, test_align_losses, kl_betas, combined_silhouettes, \
            full_combined_silhouettes, all_per_encoder_silhouettes, avg_encoder_silhouettes, nn_encoder_silhouettes, \
            test_losses_all = train_test_epoch(model, train_loader, test_loader, epoch, 
            total_annealing_steps, device, args, optimizer, 
            train_losses, test_losses, train_align_losses, test_align_losses, kl_betas, combined_silhouettes, \
            full_combined_silhouettes, all_per_encoder_silhouettes, avg_encoder_silhouettes, nn_encoder_silhouettes, \
            test_losses_all)  

            # Save model checkpoint every checkpoint_every epochs
            checkpoint_every = getattr(args, 'checkpoint_every', 0)
            if checkpoint_every > 0 and epoch % checkpoint_every == 0 and epoch < args.epochs:
                save_path = f'{RESULTS_DIR}/medvae_{args.save_name}_{args.latent_dim}_b{args.KL_beta}_epoch{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_losses[-1][0],
                    'test_loss': test_losses[-1][0],
                    'silhouette': combined_silhouettes[-1],
                    'config': vars(args)
                }, save_path)
                print(f"Checkpoint saved to {save_path}")
    # After all epochs have completed:
    # Prepare lists to hold the silhouette scores for each encoder
    silhouette_enc = [[] for _ in range(len(model.encoders))]

    # Create a dictionary to hold all loss data and metrics
    loss_data = {
        'epoch': np.arange(1, args.epochs + 1),
        'train_loss': [t[0] for t in train_losses],
        'test_loss': [t[0] for t in test_losses],
        'train_nn_rec_loss': [t[1] for t in train_losses],
        'test_nn_rec_loss': [t[1] for t in test_losses],
        'train_fmri_rec_loss': [t[2] for t in train_losses],
        'test_fmri_rec_loss': [t[2] for t in test_losses],
        'train_KLD': [t[3] for t in train_losses],
        'test_KLD': [t[3] for t in test_losses],
        'train_alignment_losses': train_align_losses, 
        'test_alignment_losses': test_align_losses,
        'kl_beta': kl_betas,
        'silhouette_fmri_only': combined_silhouettes,
        'silhouette_full': full_combined_silhouettes,
        'avg_encoder_silhouette': avg_encoder_silhouettes,
        'nn_encoder_silhouette': nn_encoder_silhouettes
    }

    # Add the all-encoder metrics to the loss_data dictionary
    if test_losses_all:  # Check if the list has items, not if it's None
        loss_data['test_loss_all'] = [t[0] for t in test_losses_all]
        loss_data['test_nn_rec_loss_all'] = [t[1] for t in test_losses_all]
        loss_data['test_fmri_rec_loss_all'] = [t[2] for t in test_losses_all]
        loss_data['test_kld_loss_all'] = [t[3] for t in test_losses_all]


    if args.train:
        # Add individual encoder silhouette scores
        for i in range(len(model.encoders)):
            # Extract the silhouette score for this encoder from each epoch's results
            encoder_silhouettes = []
            for epoch_silhouettes in all_per_encoder_silhouettes:
                if i < len(epoch_silhouettes):
                    encoder_silhouettes.append(epoch_silhouettes[i])
                else:
                    encoder_silhouettes.append(np.nan)  # For epochs where this encoder wasn't evaluated

            # Add to the dictionary
            loss_data[f'silhouette_enc{i}'] = encoder_silhouettes

        # Create directory for saving results if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)

        loss_df = pd.DataFrame(loss_data)

        _losses_csv = os.path.join(RESULTS_DIR, f'medvae_{args.save_name}_{args.latent_dim}_{args.KL_beta}_losses.csv')
        loss_df.to_csv(_losses_csv, index=False)
        print(f"Loss data saved to {_losses_csv}")

        # Also save the final model
        save_path = os.path.join(RESULTS_DIR, f'medvae_{args.save_name}_{args.latent_dim}_b{args.KL_beta}.pt')
        if args.finetune and args.load_name:
            # Save only finetuned (unfrozen) parameters
            trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
            finetuned_state_dict = {k: v for k, v in model.state_dict().items()
                                    if k in trainable_names}
            torch.save({
                'epoch': args.epochs,
                'model_state_dict': finetuned_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses[-1][0],
                'test_loss': test_losses[-1][0],
                'silhouette': combined_silhouettes[-1],
                'config': vars(args),
                'base_model': args.load_name,
                'finetune': True,
            }, save_path)
            print(f"Saved finetuned weights only ({len(finetuned_state_dict)} params) + base_model ref: {args.load_name}")
        else:
            torch.save({
                'epoch': args.epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses[-1][0],
                'test_loss': test_losses[-1][0],
                'silhouette': combined_silhouettes[-1],
                'config': vars(args)
            }, save_path)

if __name__ == '__main__':
    main()

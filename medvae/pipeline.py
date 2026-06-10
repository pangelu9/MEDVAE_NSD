from dataset import load_data_multiencoder
from model import HybridMultiEncoderVAE
from train import train_hybrid_vae, test_hybrid_vae


def create_vae_model(args, input_dims=None, output_dims=None):
    """Create the hybrid multi-encoder VAE (the model used in the paper)."""
    if not args.hybrid_vae:
        raise ValueError("This release only supports the hybrid model; pass --hybrid_vae.")
    return HybridMultiEncoderVAE(
        input_dims=input_dims,
        output_dims=output_dims,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        nn_output_dim=args.nn_output_dim,
        dropout_rate=args.dropout_rate,
        only_nn_encoder=args.only_nn_encoder, only_fmri_encoders=args.only_fmri_encoders,
        use_nn_decoder=args.use_nn_decoder, use_fmri_decoders=args.use_fmri_decoders)


def load_data(args, kwargs, gradual_intro_enc=False):
    """Load the multi-encoder data (train/test loaders + dims)."""
    return load_data_multiencoder(args, kwargs, gradual_intro_enc=gradual_intro_enc)


def train_model(model, train_loader, device, optimizer, epoch, args, total_annealing_steps=0, disabled_encoders=None):
    """Train one epoch of the hybrid VAE."""
    return train_hybrid_vae(model, train_loader, device, optimizer, epoch, args, total_annealing_steps,
                            disabled_encoders=disabled_encoders)


def test_model(model, test_loader, device, epoch, args, kl_beta, disabled_encoders=None):
    """Evaluate the hybrid VAE."""
    return test_hybrid_vae(model, test_loader, device, epoch, args, kl_beta, disabled_encoders=disabled_encoders)


def print_epoch_summary(epoch, train_loss, train_rec_loss, train_kld_loss, test_loss, test_rec_loss, test_kld_loss, fmri_combined_silhouette, avg_encoder_silhouette, args):
    # Print epoch summary
    print(f"Epoch {epoch}/{args.epochs}")
    print(f"  Train Loss: {train_loss:.4f} (Rec: {train_rec_loss:.4f}, KLD: {train_kld_loss:.4f})")
    print(f"  Test Loss: {test_loss:.4f} (Rec: {test_rec_loss:.4f}, KLD: {test_kld_loss:.4f})")
    print(f"  Combined Silhouette: {fmri_combined_silhouette:.4f}")
    print(f"  Avg Encoder Silhouette: {avg_encoder_silhouette:.4f}")


def train_test_epoch(model, train_loader, test_loader, epoch, total_annealing_steps, device, args, optimizer,
                     train_losses, test_losses, train_align_losses, test_align_losses, kl_betas, combined_silhouettes,
                     full_combined_silhouettes, all_per_encoder_silhouettes, avg_encoder_silhouettes, nn_encoder_silhouettes,
                     test_losses_all=None):

    train_loss, train_nn_rec_loss, train_fmri_rec_loss, train_kld_loss, align_loss_train, kl_beta = train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            total_annealing_steps=total_annealing_steps
        )

    # Evaluate - always expecting the additional all-encoder metrics
    test_loss, test_nn_rec_loss, test_fmri_rec_loss, test_kld_loss, align_loss_test, \
    fmri_combined_silhouette, full_combined_silhouette, avg_encoder_silhouette, \
    per_encoder_silhouettes, nn_encoder_silhouette, \
    test_loss_all, test_nn_rec_loss_all, test_fmri_rec_loss_all, test_kld_loss_all = test_model(
                model=model,
                test_loader=test_loader,
                device=device,
                epoch=epoch,
                args=args,
                kl_beta=kl_beta
            )
    # Always append to test_losses_all
    test_losses_all.append((test_loss_all, test_nn_rec_loss_all, test_fmri_rec_loss_all, test_kld_loss_all))

    # Store metrics
    train_losses.append((train_loss, train_nn_rec_loss, train_fmri_rec_loss, train_kld_loss))
    test_losses.append((test_loss, test_nn_rec_loss, test_fmri_rec_loss, test_kld_loss))
    train_align_losses.append(align_loss_train)
    test_align_losses.append(align_loss_test)
    kl_betas.append(kl_beta)
    combined_silhouettes.append(fmri_combined_silhouette)
    full_combined_silhouettes.append(full_combined_silhouette)
    all_per_encoder_silhouettes.append(per_encoder_silhouettes)
    avg_encoder_silhouettes.append(avg_encoder_silhouette)
    nn_encoder_silhouettes.append(nn_encoder_silhouette)

    print_epoch_summary(epoch, train_loss, train_nn_rec_loss, train_kld_loss, test_loss, test_nn_rec_loss, test_kld_loss, fmri_combined_silhouette, avg_encoder_silhouette, args)

    # Print additional all-encoder summary if available
    if args.eval_only_fmri and len(test_losses_all) > 0:
        print_all_encoders_summary(epoch, test_losses_all[-1])

    return train_losses, test_losses, train_align_losses, test_align_losses, kl_betas, combined_silhouettes, full_combined_silhouettes, all_per_encoder_silhouettes, avg_encoder_silhouettes, nn_encoder_silhouettes, test_losses_all


def print_all_encoders_summary(epoch, test_loss_tuple):
    """Print summary of metrics using all encoders"""
    test_loss_all, test_nn_rec_loss_all, test_fmri_rec_loss_all, test_kld_loss_all = test_loss_tuple

    print(f"  All Encoders Summary (including NN encoder):")
    print(f"  Test Loss (all): {test_loss_all:.4f}")
    print(f"  Test NN Rec Loss (all): {test_nn_rec_loss_all:.4f}")
    print(f"  Test fMRI Rec Loss (all): {test_fmri_rec_loss_all:.4f}")
    print(f"  Test KLD Loss (all): {test_kld_loss_all:.4f}")

"""Command-line interface for MED-VAE training (``main.py``).

The argument parser lives here so ``main.py`` reads as training logic rather than
~90 lines of flag declarations. ``parse_args()`` is what ``main.py`` calls;
``build_parser()`` is exposed separately for tests / programmatic use.
"""
import argparse


def build_parser():
    """Build the MED-VAE training argument parser."""
    parser = argparse.ArgumentParser(description='Multi-Encoder VAE')
    parser.add_argument('--hybrid_vae', action='store_true', default=False,
                    help='Train VAE with both NN and fMRI decoders')
    parser.add_argument('--nn_weight', type=float, default=1.0,
                    help='Weight for NN reconstruction loss (fMRI→ANN pathway)')
    parser.add_argument('--ann_pathway_weight', type=float, default=None,
                    help='Weight for ANN→ANN reconstruction loss (defaults to nn_weight if not set)')
    parser.add_argument('--fmri_weight', type=float, default=1.0,
                    help='Weight for fMRI reconstruction loss')
    parser.add_argument('--fmri_pathway_weight', type=float, default=None,
                    help='Weight for fMRI→fMRI reconstruction loss (defaults to fmri_weight if not set)')



    parser.add_argument('--load_name', type=str, default=None, metavar='N',
                    help='name of model to be loaded for comparison')

    parser.add_argument('--remove_overlaps', action='store_true', default=False,
                   help='remove_overlaps in fmri data')
    parser.add_argument('--eval_only_fmri', action='store_true', default=False,
                   help='Evaluate using only fMRI encoders during testing')
    parser.add_argument('--save_name', type=str, default="", metavar='N',
                    help='name of model to be saved')
    parser.add_argument('--checkpoint_every', type=int, default=0,
                    help='Save checkpoint every N epochs (0 = disabled)')
    parser.add_argument('--dataset', type=str, default="algonauts", metavar='N',
                    help='dataset: algonauts/streams')
    parser.add_argument('--only_nn_encoder', action='store_true', default=False,
                    help='Use only the NN activations encoder (encoder #8)')
    parser.add_argument('--only_fmri_encoders', action='store_true', default=False,
                    help='Use only the fMRI encoders (encoder #1-#7)')
    parser.add_argument('--use_nn_decoder', action='store_true', default=False,
                   help='Use NN decoder in hybrid VAE')
    parser.add_argument('--use_fmri_decoders', action='store_true', default=False,
                    help='Use fMRI decoders in hybrid VAE')
    parser.add_argument('--nn_to_fmri', action='store_true', default=False,
                   help='Enable NN encoder → fMRI decoders path in hybrid model')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--KL_beta', type=float, default=0.1, help='weight for KL term in loss')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension size')
    parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension size')
    parser.add_argument('--output_dim', type=int, nargs="+", default=[20732, 20735, 20736, 20733, 20733, 20734, 20726, 20733], help='output dimension size')
    parser.add_argument('--nn_output_dim', type=int, default=7168, help='nn output dimension size')
    parser.add_argument('--max_memory_gb', type=float, default=100.0,
                   help='Maximum memory to use for data loading (GB)')

    parser.add_argument('--input_dim', type=int, nargs="+",
                    default=[20732, 20735, 20736, 20733, 20733, 20734, 20726, 20733, 7168],
                    help='input dimension sizes')
    parser.add_argument('--keep_percent', type=float, nargs="+",
                    default=[100, 100, 100, 100, 100, 100, 100, 100],
                    help='what percent of images to kepp for each subject [sparsity]')
    parser.add_argument('--test_size', type=float, default=0.1, help='test set size ratio')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--kl_annealing', type=str, default='none', choices=['none', 'cyclical', 'monotonic'], help='KL divergence annealing strategy')
    parser.add_argument('--noise_level', type=float, default=0.0, help='level of noise to add to fMRI data')
    parser.add_argument('--framework', type=str, default='multiencoder', help='framework type')
    # Add arguments that might be required by the imported load_labels and load_activations functions
    parser.add_argument('--brain_to_brain', action='store_true', default=False, help='brain to brain mode')
    parser.add_argument('--all_subjects', action='store_true', default=True, help='use all subjects')
    parser.add_argument('--shuffle_fmri', action='store_true', default=False, help='shuffle fMRI data')
    parser.add_argument('--model_to_brain', action='store_true', default=False, help='model to brain mode')
    parser.add_argument('--model_to_model', action='store_true', default=False, help='model to model mode')
    parser.add_argument('--train', action='store_true', default=False, help='train model')
    parser.add_argument('--filename', type=str, help='activations filename')

    parser.add_argument('--weight_annealing', type=str, default=None,
                        choices=['fmri', 'nn'],
                        help='Type of weight annealing: "fmri" (increase from 0 to fmri_weight) or "nn" (decrease from nn_weight to 1)')

    parser.add_argument('--weight_annealing_epochs', type=int, default=None,
                        help='Number of epochs over which to perform weight annealing (default: total epochs)')

    parser.add_argument('--remove_all_overlaps', action='store_true',
                     help='Completely remove overlapping images from ALL subjects')
    parser.add_argument('--filter_no_fmri', action='store_true', default=False,
                        help='keep only NN samples that have fmri data')
    parser.add_argument('--exclude_mask', type=str, default=None,
                        help='Path to boolean .npy mask (True=exclude). Applied before train/test split.')
    parser.add_argument('--finetune', action='store_true', default=False,
                        help='Save only finetuned encoder/decoder weights (not frozen components). Requires --load_name.')
    parser.add_argument('--fmri_shuffle_voxels', action='store_true', default=False,
                        help='permute fmri voxels for each sample, in the same way for all')
    return parser


def parse_args(argv=None):
    """Parse MED-VAE training arguments (defaults to ``sys.argv``)."""
    return build_parser().parse_args(argv)

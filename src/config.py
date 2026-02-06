"""
config.py

This module contains model configuration and argument parsing.

Functions:
- parse_args: Parse command line arguments for model training and evaluation
"""

import argparse


def parse_args():
    """
    Parse command line arguments for hyperparameter configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing all configuration parameters
    """
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for UniCM model training and evaluation")

    # --- Model & Data Parameters ---
    parser.add_argument("--data_root", type=str, default='../dataset', help="Root directory for climate datasets")
    parser.add_argument("--model", type=str, default='UniCM', help="Model type identifier")
    parser.add_argument("--his_len", type=int, default=12, help="Length of historical input sequence (months)")
    parser.add_argument("--pred_len", type=int, default=24, help="Length of prediction sequence (months)")
    parser.add_argument("--interp", type=int, default=0, help="Interpolation steps (0 for no interpolation)")
    parser.add_argument("--climate_mode", type=str, default='all',  help="Specific climate mode for loss calculation")
    parser.add_argument("--mode_interaction", type=str, default='1', help="Flag for mode interaction settings")
    parser.add_argument("--resolution", type=int, default=5, help="Spatial resolution factor")
    parser.add_argument("--input_channal", type=int, default=5, help="Number of input data channels")
    parser.add_argument("--patch_size", type=str, default='2-2', help="Patch size as 'height-width' (e.g., '5-5')")
    parser.add_argument("--time_patch", type=int, default=0, help="Flag for using temporal patching")
    parser.add_argument("--t_patch_len", type=int, default=1, help="Length of temporal patches")
    parser.add_argument("--stride", type=int, default=1, help="Stride for patching")
    parser.add_argument("--training_data", type=str, default='CESM2-FV2*gr', help="Identifier for training dataset")
    parser.add_argument("--t20d_mode", type=int, default=1, help="Flag for including T20D data channel")

    # --- Transformer Architecture Parameters ---
    parser.add_argument("--d_size", type=int, default=256, help="Embedding dimension (d_model)")
    parser.add_argument("--nheads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="Dimension of the feedforward network (default: d_size * 4)")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # --- Training Parameters ---
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay (L2 regularization)")
    parser.add_argument("--min_lr", type=float, default=5e-6, help="Minimum learning rate for annealing")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr_anneal_steps", type=int, default=10000, help="Steps for learning rate annealing")
    parser.add_argument("--epochs", type=int, default=200, help="Total number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--early_stop", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--lr_early_stop", type=int, default=3, help="Patience for learning rate scheduler early stop")
    parser.add_argument("--metrics_mode", type=str, default='mean', choices = ['weight','mean'])
    parser.add_argument("--exp_folder", type=str, default='train')
    parser.add_argument("--result_filename", type=str, default='result_all.txt', help="Filename for saving results")

    # --- Loss Coefficients ---
    parser.add_argument("--lambda1", type=float, default=1.0, help="Coefficient for the main reconstruction loss")
    parser.add_argument("--lambda3", type=float, default=1.0, help="Coefficient for the climate mode skill loss")
    parser.add_argument("--lambda2", type=float, default=0.0, help="Coefficient for the explicit mode prediction loss")
    parser.add_argument("--loss_nino", type=int, default=1, help="Flag to include specific Nino loss")
    parser.add_argument("--loss_all", type=int, default=0, help="Flag for alternative loss calculation")

    # --- Environment & IO ---
    parser.add_argument("--mode", type=str, default='training', choices=['testing','training'])
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA device ID to use")
    parser.add_argument("--pretrained_path", type=str, default='', help="Path to pretrained model for evaluation/bagging")
    parser.add_argument("--pretrained_simulate_path", type=str, default='', help="Path to a pretrained simulation model")
    parser.add_argument("--log_interval", type=int, default=20, help="Step interval for logging training loss")
    parser.add_argument("--val_interval", type=int, default=1, help="Epoch interval for running validation")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    
    # --- Miscellaneous ---
    parser.add_argument("--norm_std", type=int, default=1, help="Flag for normalization strategy")
    parser.add_argument("--explore", type=int, default=1, help="Exploration flag")
    parser.add_argument("--sv_ratio", type=float, default=0.0, help="SV ratio")
    parser.add_argument("--autoregressive", type=int, default=0, help="Flag for autoregressive prediction")
    parser.add_argument("--num_ensemble", type=int, default=5, help="Number of ensemble models for averaging predictions")

    args = parser.parse_args()
    return args

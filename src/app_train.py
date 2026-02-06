"""
app_train.py

Model training script for UniCM climate model.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from LoadData import *
from Trainer import TrainLoop
from models import UniCM
import copy
from settings import setup_init, setup_training_environment
from config import parse_args

from numba import cuda

torch.autograd.set_detect_anomaly(True)


def main(mypara):
    """
    Main training function.
    """
    # Setup training environment
    device = setup_training_environment(mypara)

    """
    Data preparation
    """
    train_data, val_data = create_training_dataloaders(mypara)

    """
    Model initialization
    """
    mymodel = UniCM(mypara).to(device)

    model_size = sum(p.numel() * p.element_size() for p in mymodel.parameters()) 
    model_size_mb = model_size / (1024 ** 2)

    print(f"\nModel size: {model_size_mb:.2f} MB\n")

    """
    Model training
    """
    TrainLoop(
        args = mypara,
        model=mymodel,
        train_data= train_data,
        val_data = val_data,
        device=device,
    ).run_loop()


if __name__ == "__main__":
    mypara = parse_args()
    setup_init(mypara.seed)
    main(mypara)
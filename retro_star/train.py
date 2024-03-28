"""
This script is used to train a ValueMLP model for retrosynthetic planning in chemistry. 
The ValueMLP model is a neural network model that is used as a value function to guide the search 
in the retrosynthetic planning process. 

The script defines a function `train` that prepares the model and the data loaders, 
creates a Trainer instance, and starts the training process. 

If the script is run directly, it sets the random seeds for reproducibility, sets up the logger, 
and calls the `train` function.
"""

import os
import numpy as np
import torch
import random
import pickle
import torch.nn.functional as F
import logging
from retro_star.common import args
from retro_star.model import ValueMLP
from retro_star.data_loader import ValueDataLoader
from retro_star.trainer import Trainer
from retro_star.utils import setup_logger

def train():
    # Set the device for torch
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    # Create the ValueMLP model
    model = ValueMLP(
        n_layers=args.n_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1,
        device=device
    )

    # Check if the training data file exists
    assert os.path.exists('%s/%s.pt' % (args.value_root, args.value_train))

    # Create the data loader for the training data
    train_data_loader = ValueDataLoader(
        fp_value_f='%s/%s' % (args.value_root, args.value_train),
        batch_size=args.batch_size
    )

    # Create the data loader for the validation data
    val_data_loader = ValueDataLoader(
        fp_value_f='%s/%s' % (args.value_root, args.value_val),
        batch_size=args.batch_size
    )

    # Create the Trainer instance
    trainer = Trainer(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        save_epoch_int=args.save_epoch_int,
        model_folder=args.save_folder,
        device=device
    )

    # Start the training process
    trainer.train()


if __name__ == '__main__':
    # Set the random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Set up the logger
    setup_logger('train.log')

    # Call the train function
    train()
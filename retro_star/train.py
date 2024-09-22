import os
import numpy as np
import torch
import random
import pickle
import torch.nn.functional as F
import logging
from common import args
from model import ValueMLP
from data_loader import ValueDataLoader
from trainer import Trainer
from utils import setup_logger

def train():
    """
    Trains the ValueMLP model using the provided training and validation data.

    Returns:
        None
    """
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate, device):
        """
         Initialize the MLP. This is the method that will be called by the constructor. 
         
         @param n_layers: Number of layers in the model
         @param fp_dim: Dimension of the predictor layer ( 1D )
         @param latent_dim: Dimension of the latent variable ( 1D )
         @param dropout_rate: Dropout rate in Hopcroft
         @param device: Device to run the model on e. g
        """
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.device = device

        logging.info('Initializing value model: latent_dim=%d' % self.latent_dim)

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        # Creates a new layer with the same shape as the model.
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        """
         Forward pass of the model.
         
         @param fps: ( Tensor ) Frame rate
         @return: ( Tensor ) Output of the model as a batch of feature maps each of which is a 2D
        """
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))

        return x

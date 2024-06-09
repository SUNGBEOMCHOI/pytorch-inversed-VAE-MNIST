import numpy as np
import torch 
import torch.nn as nn

from utils import build_network

class Model(nn.Module):
    def __init__(self, model_cfg, device=torch.device('cpu')):
        """
        Variational Autoencoder (VAE) model.

        Args:
            model_cfg (dict): Dictionary containing model configuration.
            device (torch.device, optional): Torch device to use (CPU or GPU). Default is CPU.
        """
        super().__init__()
        self.device = device
        architecture_encoder = model_cfg['architecture']['encoder']
        architecture_decoder = model_cfg['architecture']['decoder']

        self.encoder_conv = build_network(architecture_encoder['conv'])
        self.encoder_mean = build_network(architecture_encoder['mean'])
        self.encoder_log_var = build_network(architecture_encoder['log_var'])

        self.decoder = build_network(architecture_decoder)

        self.mean, self.log_var, self.output = None, None, None

    def forward(self, x):
        """
        Encode input images and decode the latent vector with added noise.

        Args:
            x (numpy.ndarray or torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Decoded output images.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        self.mean, self.log_var = self.encoding(x)
        z = self.add_noise(self.mean, self.log_var)
        output = self.decoding(z)
        return output

    def encoding(self, x):
        """
        Encode input images to a Gaussian distribution.

        Args:
            x (torch.Tensor): Batch of input images.

        Returns:
            tuple: Mean and log variance of the encoded images.
        """
        x = x.to(self.device)
        x = x.reshape(x.shape[0], -1)
        x = self.encoder_conv(x)
        self.mean = self.encoder_mean(x)
        self.log_var = self.encoder_log_var(x)
        return self.mean, self.log_var

    def decoding(self, x):
        """
        Decode latent vectors to images.

        Args:
            x (torch.Tensor): Batch of latent vectors.

        Returns:
            torch.Tensor: Decoded batch of images.
        """
        x = x.to(self.device)
        x = self.decoder(x)
        self.output = x.reshape(x.shape[0], 1, 28, 28)
        return self.output

    def add_noise(self, mean, log_var):
        """
        Generate random values from a Gaussian distribution using the given mean and log variance.

        Args:
            mean (torch.Tensor): Mean of the Gaussian distribution.
            log_var (torch.Tensor): Log variance of the Gaussian distribution.

        Returns:
            torch.Tensor: Random values sampled from the Gaussian distribution.
        """
        noise = torch.normal(0, 1, size=log_var.shape, device=self.device)
        std = torch.exp(0.5 * log_var)
        return mean + std * noise
    
class DeterministicModel(nn.Module):
    def __init__(self, model_cfg, device=torch.device('cpu')):
        """
        Variational Autoencoder (VAE) model without noise.

        Args:
            model_cfg (dict): Dictionary containing model configuration.
            device (torch.device, optional): Torch device to use (CPU or GPU). Default is CPU.
        """
        super().__init__()
        self.device = device
        architecture_encoder = model_cfg['architecture']['encoder']
        architecture_decoder = model_cfg['architecture']['decoder']

        self.encoder = build_network(architecture_encoder['conv'])
        self.decoder = build_network(architecture_decoder)

    def forward(self, x):
        """
        Encode input images and decode the latent vector.

        Args:
            x (numpy.ndarray or torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Decoded output images.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
        latent_vector = self.encoding(x)
        output = self.decoding(latent_vector)
        return output

    def encoding(self, x):
        """
        Encode input images to a latent vector.

        Args:
            x (torch.Tensor): Batch of input images.

        Returns:
            torch.Tensor: Latent vector of the encoded images.
        """
        x = x.to(self.device)
        x = x.reshape(x.shape[0], -1)
        x = self.encoder(x)
        return x

    def decoding(self, x):
        """
        Decode latent vectors to images.

        Args:
            x (torch.Tensor): Batch of latent vectors.

        Returns:
            torch.Tensor: Decoded batch of images.
        """
        x = x.to(self.device)
        x = self.decoder(x)
        output = x.reshape(x.shape[0], 1, 28, 28)
        return output
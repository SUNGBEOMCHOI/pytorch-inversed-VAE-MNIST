import copy
import collections

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

from inverse_model import InverseDecoderModel

def loss_func(loss_name_list):
    """
    Return a list of loss functions based on provided names.

    Args:
        loss_name_list (list of str): List of loss function names.

    Returns:
        list: List of corresponding PyTorch loss functions.
    """
    loss_func_list = []
    for loss_name in loss_name_list:
        loss_name = loss_name.lower()
        if loss_name == 'mseloss':
            loss = nn.MSELoss()
        elif loss_name == 'crossentropyloss':
            loss = nn.CrossEntropyLoss()
        elif loss_name == 'huberloss':
            loss = nn.HuberLoss()
        elif loss_name == 'kldivloss':
            loss = nn.KLDivLoss()
        elif loss_name == 'bceloss':
            loss = nn.BCELoss()
        elif loss_name == 'regularizationloss':
            loss = regularization_loss
        else:
            raise NotImplementedError(f"Loss function {loss_name} is not implemented.")
        loss_func_list.append(loss)
    return loss_func_list

def regularization_loss(mean, log_var):
    """
    Compute the regularization loss for a Variational Autoencoder (VAE).

    Args:
        mean (torch.Tensor): Tensor containing the means.
        log_var (torch.Tensor): Tensor containing the logarithm of the variances.

    Returns:
        torch.Tensor: The computed regularization loss.
    """
    return torch.mean(0.5 * (torch.pow(mean, 2) - log_var + torch.exp(log_var) - 1))

def optim_func(model, cfg):
    """
    Return a PyTorch optimizer based on the provided configuration.

    Args:
        model (torch.nn.Module): The model to be optimized.
        cfg (dict): Dictionary containing optimizer configuration, with keys 'name' (str), 'learning_rate' (float), and 'others' (dict).

    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    optim_name = cfg['name'].lower()
    learning_rate = cfg['learning_rate']
    others = cfg['others'] if cfg['others'] else {}
    if optim_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, **others)
    elif optim_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, **others)
    elif optim_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, **others)
    else:
        raise NotImplementedError(f"Optimizer {optim_name} is not implemented.")
    return optimizer

def lr_scheduler_func(optimizer, cfg):
    """
    Return a PyTorch learning rate scheduler based on the provided configuration.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        cfg (dict): Dictionary containing learning rate scheduler configuration, with keys 'name' (str) and 'others' (dict).

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Configured learning rate scheduler.
    """
    scheduler_name = cfg['name'].lower()
    others = cfg['others'] if cfg['others'] else {}
    if scheduler_name == 'steplr':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **others)
    elif scheduler_name == 'multisteplr':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **others)
    elif scheduler_name == 'cosineannealinglr':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **others)
    elif scheduler_name == 'cycliclr':
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, **others)
    elif scheduler_name == 'lambdalr':
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, **others)
    else:
        raise NotImplementedError(f"Scheduler {scheduler_name} is not implemented.")
    return lr_scheduler

def plot_progress(history, epoch, file_path='./train_progress'):
    """
    Plot and save the training and validation loss progress.

    Args:
        history (dict): Dictionary containing 'train' and 'validation' loss history.
        epoch (int): The current training epoch.
        file_path (str, optional): Path to save the plot. Default is './train_progress'.

    Returns:
        None
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, epoch+1, dtype=np.int16), history['train'], label='Train Loss')
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, epoch+1, dtype=np.int16), history['validation'], label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'{file_path}/{epoch}_epochs.png')
    plt.close()

class Builder(object):
    """
    Builder for creating PyTorch models from a configuration file.

    Args:
        namespaces (iterable of dict): A list of namespaces (dictionaries) containing available components.
    """
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)

    def __call__(self, name, *args, **kwargs):
        """
        Instantiate a component by name using provided arguments.

        Args:
            name (str): The name of the component to instantiate.
            *args: Positional arguments to pass to the component's constructor.
            **kwargs: Keyword arguments to pass to the component's constructor.

        Returns:
            object: Instantiated component.
        """
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e

    def add_namespace(self, namespace, index=-1):
        """
        Add a new namespace to the builder's chain of namespaces.

        Args:
            namespace (dict): The namespace to add.
            index (int, optional): The position to insert the namespace. Default is -1 (add at the end).

        Returns:
            None
        """
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)

def build_network(architecture, builder=Builder(torch.nn.__dict__)):
    """
    Build a neural network from a configuration.

    Args:
        architecture (list of dict): List of dictionaries representing network layers and their configurations.
        builder (Builder, optional): Builder object to create layers. Default is Builder(torch.nn.__dict__).

    Returns:
        torch.nn.Sequential: Sequential model built from the configuration.
    """
    layers = []
    architecture = copy.deepcopy(architecture)
    for block in architecture:
        assert len(block) == 1
        name, kwargs = list(block.items())[0]
        if kwargs is None:
            kwargs = {}
        args = kwargs.pop("args", [])
        layers.append(builder(name, *args, **kwargs))
    return torch.nn.Sequential(*layers)

def save_model(epoch, model, optimizer, history, lr_scheduler, file_path='./pretrained'):
    """
    Save the training checkpoint.

    Args:
        epoch (int): The current training epoch.
        model (torch.nn.Module): The trained model.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler used in training.
        file_path (str, optional): Path to save the checkpoint. Default is './pretrained'.

    Returns:
        None
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }, f'{file_path}/model_{epoch}.pt')

def get_train_dataset():
    """
    Return the training and validation datasets.

    In this case, it returns the MNIST training dataset split into training and validation sets.

    Returns:
        tuple: Tuple containing the training dataset and validation dataset.
    """
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    train_size = int(len(train_dataset) * 0.9)
    validation_size = len(train_dataset) - train_size

    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])

    return train_dataset, validation_dataset

def get_test_dataset():
    """
    Return the test dataset.

    In this case, it returns the MNIST test dataset.

    Returns:
        torch.utils.data.Dataset: The test dataset.
    """
    test_dataset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
    return test_dataset


def get_dataloader(dataset, batch_size, train=True):
    """
    Return a torch DataLoader for training or validation.

    Args:
        dataset: Torch dataset object.
        batch_size (int): The number of samples per batch to load.
        train (bool, optional): If True, return a DataLoader for the training phase with shuffling and dropping the last incomplete batch.
                                If False, return a DataLoader for the validation/test phase without shuffling and without dropping the last batch.
                                Default is True.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the provided dataset.
    """
    if train:
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=8)

def convert_decoder_params_to_numpy(model):
    """
    Convert the parameters of the decoder layers of a model to NumPy arrays.

    Args:
        model (torch.nn.Module): The PyTorch model containing decoder layers.

    Returns:
        dict: A dictionary where keys are parameter names and values are NumPy arrays of the corresponding parameters.
    """
    decoder_params_numpy = {}
    for name, param in model.named_parameters():
        if 'decoder' in name:
            decoder_params_numpy[name] = param.detach().cpu().numpy()
    return decoder_params_numpy

def compute_pseudo_inverse(params_numpy):
    """
    Compute the pseudo-inverse of weight matrices in the given parameters.

    Args:
        params_numpy (dict): Dictionary of model parameters in NumPy array format.

    Returns:
        dict: A dictionary with the same keys as input, where weight matrices are replaced with their pseudo-inverses and other parameters remain unchanged.
    """
    pseudo_inverse_params = {}
    for name, param in params_numpy.items():
        if 'weight' in name:  # Only apply to weight matrices
            pseudo_inverse_params[name] = np.linalg.pinv(param)
        else:
            pseudo_inverse_params[name] = param  # Biases are not inverted
    return pseudo_inverse_params

def create_encoder_from_decoder_pinv(pseudo_inverse_params):
    """
    Create encoder parameters from the pseudo-inverse of decoder parameters.

    Args:
        pseudo_inverse_params (dict): Dictionary of pseudo-inversed decoder parameters.

    Returns:
        dict: A dictionary of encoder parameters constructed from the pseudo-inversed decoder parameters.
    """
    encoder_params = {}
    decoder_layers = sorted(pseudo_inverse_params.keys(), reverse=True, key=lambda x: int(x.split('.')[1]))
    
    for name in decoder_layers:
        layer_idx = int(name.split('.')[1])
        new_layer_idx = len(decoder_layers) - 1 - layer_idx
        
        new_name = name.replace(f'decoder.{layer_idx}', f'encoder.{new_layer_idx}')
        encoder_params[new_name] = pseudo_inverse_params[name]
    
    return encoder_params

def create_inverse_decoder_model(decoder_model):
    """
    Create an inverse decoder model from a given decoder model.

    Args:
        decoder_model (torch.nn.Module): The original PyTorch decoder model.

    Returns:
        InverseDecoderModel: A new model instance with parameters created from the pseudo-inverse of the decoder model's parameters.
    """
    decoder_params_numpy = convert_decoder_params_to_numpy(decoder_model)
    pseudo_inverse_params = compute_pseudo_inverse(decoder_params_numpy)
    encoder_params = create_encoder_from_decoder_pinv(pseudo_inverse_params)
    return InverseDecoderModel(encoder_params)

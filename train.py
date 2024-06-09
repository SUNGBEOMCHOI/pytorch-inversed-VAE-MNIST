import os
import argparse

import yaml
import torch

from models import Model, DeterministicModel
from utils import loss_func, optim_func, lr_scheduler_func, plot_progress, save_model, get_train_dataset, get_dataloader

def train(args, cfg):
    """
    Train a Variational Autoencoder (VAE) model.

    Args:
        args: Additional arguments.
        cfg (dict): Configuration dictionary containing training parameters and model settings.
    
    Returns:
        None
    """
    ########################
    #   Get configuration  #
    ########################
    device = torch.device('cuda' if cfg['device'] == 'cuda' and torch.cuda.is_available() else ('mps' if cfg['device'] == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'))
    train_cfg = cfg['train']
    batch_size = train_cfg['batch_size']
    train_epochs = train_cfg['train_epochs']
    loss_name_list = train_cfg['loss']
    optim_cfg = train_cfg['optim']
    lr_scheduler_cfg = train_cfg['lr_scheduler']
    alpha = train_cfg['alpha']
    model_path = train_cfg['model_path']
    progress_path = train_cfg['progress_path']
    plot_epochs = train_cfg['plot_epochs']
    model_cfg = cfg['model']

    ########################
    #      Make model      #
    ########################
    # model = Model(model_cfg, device).to(device)
    model = DeterministicModel(model_cfg, device).to(device)
    print(model)

    ########################
    #    Train settings    #
    ########################
    train_dataset, valid_dataset = get_train_dataset()
    train_loader = get_dataloader(train_dataset, batch_size, train=True)
    valid_loader = get_dataloader(valid_dataset, batch_size, train=False)
    criterion_list = loss_func(loss_name_list)
    # reconstruction_criterion, regularization_criterion = criterion_list
    reconstruction_criterion = criterion_list[0]
    optimizer = optim_func(model, optim_cfg)
    lr_scheduler = lr_scheduler_func(optimizer, lr_scheduler_cfg)
    history = {'train': [], 'validation': []}  # for saving loss
    start_epoch = 1

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(progress_path, exist_ok=True)

    ########################
    #      Train model     #
    ########################
    for epoch in range(start_epoch, train_epochs + 1):
        total_loss = 0.0
        model.train()
        for input, _ in train_loader:
            input, target = input.to(device), input.to(device)
            # mean, log_var = model.encoding(input.detach())
            z = model.encoding(input)
            # z = model.add_noise(mean, log_var)
            output = model.decoding(z)

            reconstruction_loss = reconstruction_criterion(output, target.detach())
            # regularization_loss = regularization_criterion(mean, log_var)
            # loss = reconstruction_loss + alpha * regularization_loss
            loss = reconstruction_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)
        history['train'].append(total_loss)
        validation_loss = validation(model, valid_loader, criterion_list, alpha, device)
        history['validation'].append(validation_loss)
        print(f'------ {epoch:03d} training ------- train loss : {total_loss:.6f} -------- validation loss : {validation_loss:.6f} -------')
        if epoch % plot_epochs == 0:
            plot_progress(history, epoch, progress_path)
            save_model(epoch, model, optimizer, history, lr_scheduler, model_path)
        lr_scheduler.step()

def validation(model, validation_loader, criterion_list, alpha, device):
    """
    Validate a trained model.

    Args:
        model (torch.nn.Module): Trained model to validate.
        validation_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion_list (list): List of PyTorch loss functions.
        alpha (float): Weight for the regularization loss.
        device (torch.device): Device to perform validation on (CPU or GPU).

    Returns:
        float: Total validation loss.
    """
    total_loss = 0.0
    # reconstruction_criterion, regularization_criterion = criterion_list
    reconstruction_criterion = criterion_list[0]
    model.eval()

    for input, _ in validation_loader:
        input, target = input.to(device), input.to(device)
        with torch.no_grad():
            # mean, log_var = model.encoding(input)
            z = model.encoding(input)
            # z = model.add_noise(mean, log_var)
            output = model.decoding(z)

            reconstruction_loss = reconstruction_criterion(output.detach(), target.detach())
            # regularization_loss = regularization_criterion(mean.detach(), log_var.detach())
            # loss = reconstruction_loss + alpha * regularization_loss
            loss = reconstruction_loss
        total_loss += loss.item()
    total_loss /= len(validation_loader)
    
    return total_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/deterministic_config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    train(args, cfg)
import os
import copy
import yaml
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from models import Model
from utils import get_test_dataset, get_dataloader, create_inverse_decoder_model

def test(args, cfg):
    """
    Test inversed decoder model
    
    It contains a test phase
    1. Distribution of latent vector
        Plot encoded latent vector of input images
    """
    ########################
    #   Get configuration  #
    ########################
    device = torch.device('cuda' if cfg['device'] == 'cuda' and torch.cuda.is_available() else ('mps' if cfg['device'] == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'))
    test_cfg = cfg['test']
    n_components = test_cfg['n_components'] # components number of dimension reduction
    batch_size = test_cfg['batch_size']
    results_path = test_cfg['results_path']
    model_cfg = cfg['model']

    ########################
    # Get pretrained model #
    ########################
    pretrained_model = Model(model_cfg, device).to(device)
    checkpoint = torch.load(test_cfg['model_path'], map_location=device)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])

    print("Model structure:")
    print(pretrained_model)

    test_dataset = get_test_dataset()
    test_loader = get_dataloader(test_dataset, batch_size, train=False)
    embedded_list, target_list = [], []

    os.makedirs(results_path, exist_ok=True)

    ########################
    #  Get inversed model  #
    ########################
    model = create_inverse_decoder_model(pretrained_model).to(device)

    print("\nInverse model structure:")
    print(model)

    ########################
    #      Test model      #
    ########################
    model.eval()
    
    sne_model = TSNE(n_components=n_components) # For dimension reduction
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
        embedded_list.extend(copy.deepcopy(outputs.cpu().numpy()))
        target_list.extend(copy.deepcopy(targets.to(torch.int8).cpu().numpy()))

    X_embedded = sne_model.fit_transform(np.array(embedded_list)) # T-SNE

    #################################
    # Distribution of latent vector #
    #################################
    palette = sns.color_palette("bright", 10)
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=target_list, legend='full', palette=palette)
    plt.savefig(f'./{results_path}/embedding.png')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/inverse_config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
       
    test(args, cfg)

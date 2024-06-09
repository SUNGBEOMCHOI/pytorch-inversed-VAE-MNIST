import os
import argparse
import collections
import copy

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from models import Model, DeterministicModel
from utils import get_test_dataset, get_dataloader

def test(args, cfg):
    """
    Test a trained VAE model.

    This function performs three phases of evaluation:
    1. Reconstruction Evaluation: Evaluate the model's ability to reconstruct input images through encoding and decoding.
    2. Random Generation Evaluation: Evaluate the model's ability to generate images from random latent vectors.
    3. Distribution of Latent Vectors: Plot the distribution of encoded latent vectors of input images using T-SNE.

    Args:
        args: Additional arguments.
        cfg (dict): Configuration dictionary containing test parameters and model settings.

    Returns:
        None
    """
    ########################
    #   Get configuration  #
    ########################
    device = torch.device('cuda' if cfg['device'] == 'cuda' and torch.cuda.is_available() else ('mps' if cfg['device'] == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'))
    test_cfg = cfg['test']
    n_components = test_cfg['n_components']  # Number of components for dimensionality reduction
    batch_size = test_cfg['batch_size']
    results_path = test_cfg['results_path']
    model_cfg = cfg['model']
    latent_vector_dim = model_cfg['architecture']['decoder'][0]['Linear']['args'][0] # latent vector size

    ########################
    # Get pretrained model #
    ########################
    # model = Model(model_cfg, device).to(device)
    model = DeterministicModel(model_cfg, device).to(device)
    checkpoint = torch.load(test_cfg['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = get_test_dataset()
    test_loader = get_dataloader(test_dataset, batch_size, train=False)
    origin_images = collections.defaultdict(list)
    generated_images = collections.defaultdict(list)
    embedded_list, target_list = [], []

    os.makedirs(results_path, exist_ok=True)

    ########################
    #      Test model      #
    ########################
    model.eval()
    
    sne_model = TSNE(n_components=n_components)  # For dimensionality reduction
    count = 0  # Count for test iterations
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            # mean, log_var = model.encoding(inputs)
            z = model.encoding(inputs)
            outputs = model.decoding(z)
        embedded_list.extend(copy.deepcopy(z.cpu().numpy()))
        target_list.extend(copy.deepcopy(targets.to(torch.int8).cpu().numpy()))

        for idx, target in enumerate(targets):
            origin_images[int(target)].append(inputs[idx].detach().cpu().numpy()[0])
            generated_images[int(target)].append(outputs[idx].detach().cpu().numpy()[0])

    X_embedded = sne_model.fit_transform(np.array(embedded_list))  # T-SNE

    #################################
    # Distribution of latent vector #
    #################################
    palette = sns.color_palette("bright", 10)
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=target_list, legend='full', palette=palette)
    plt.savefig(f'./{results_path}/embedding.png')
    plt.close()

    #################################
    #    Reconstruction evaluation  #
    #################################
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.figure(figsize=(5, 10))
    for row in range(10):
        for column in range(10):
            origin_image = origin_images[row][column]
            generate_image = generated_images[row][column]
            plt.subplot(20, 10, 20*row + column + 1)
            plt.axis('off')
            plt.imshow(origin_image, cmap='gray')
            plt.subplot(20, 10, 20*row + column + 11)
            plt.axis('off')
            plt.imshow(generate_image, cmap='gray')
    plt.savefig(f'./{results_path}/reconstruction.png')
    plt.close()

    #################################
    # Random generation evaluation  #
    #################################
    mean = torch.randn((100, latent_vector_dim), device=device)
    with torch.no_grad():
        outputs = model.decoding(mean)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.figure(figsize=(5, 5))
    for row in range(10):
        for column in range(10):
            image = outputs[10*row + column].detach().cpu().numpy()[0]
            plt.subplot(10, 10, 10*row + column + 1)
            plt.axis('off')
            plt.imshow(image, cmap='gray')
    plt.savefig(f'./{results_path}/random_generation.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/deterministic_config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    test(args, cfg)

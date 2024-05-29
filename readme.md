
# pytorch-inversed-VAE-MNIST

## Overview

This repository contains an implementation of a Variational Autoencoder (VAE) using PyTorch, designed to explore the possibility of reconstructing inputs from outputs in deep learning models. The project examines the use of pseudo-inverses and activation function inverses in fully connected layers to achieve this goal.

For more detailed information, please refer to the blog post linked below:
[역함수와 역행렬로 딥러닝의 결과로부터 입력 복원하기](https://myinnerside.tistory.com/49)


## Description

### Objective

The main objective is to investigate whether it is possible to reconstruct the input from the output in deep learning models. The exploration started with the idea that good prompts are crucial for large language models (LLMs) and image generation models. If we could reverse-engineer the prompts from the outputs, we might better understand how to generate desired results.

### Autoencoder

An autoencoder model was trained on the MNIST dataset, using only fully connected layers with Tanh and Sigmoid activation functions, which have inverse functions. The architecture of the model is as follows:

```python
Model(
  (encoder_conv): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=False)
    (1): Tanh()
    (2): Linear(in_features=512, out_features=256, bias=False)
    (3): Tanh()
    (4): Flatten(start_dim=1, end_dim=-1)
  )
  (encoder_mean): Sequential(
    (0): Linear(in_features=256, out_features=16, bias=False)
  )
  (encoder_log_var): Sequential(
    (0): Linear(in_features=256, out_features=16, bias=False)
  )
  (decoder): Sequential(
    (0): Linear(in_features=16, out_features=256, bias=False)
    (1): Tanh()
    (2): Linear(in_features=256, out_features=512, bias=False)
    (3): Tanh()
    (4): Linear(in_features=512, out_features=784, bias=False)
    (5): Sigmoid()
  )
)
```

### Inversed Decoder Model

To attempt input reconstruction from outputs, the decoder structure was inverted. The inverted decoder uses pseudo-inverses for the fully connected layers and inverse functions for the activation functions. The architecture of the inverted decoder is as follows:

```python
(inversed decoder): Sequential(
    (0): SigmoidInverse()
    (1): Linear(in_features=784, out_features=512, bias=False)
    (2): TanhInverse()
    (3): Linear(in_features=512, out_features=256, bias=False)
    (4): TanhInverse()
    (5): Linear(in_features=256, out_features=16, bias=False)
)
```

### Results

The results show that while the model can reconstruct images to some extent, the inversion approach using pseudo-inverses and activation function inverses has limitations. The reconstructed inputs from the output did not match the original inputs well, primarily due to the constraints of the inverse functions used.

### Challenges

1. **Sigmoid Inverse Issues:** When pixel values are zero, the inverse sigmoid function can cause issues due to the logit function's behavior, leading to infinite values.
2. **Tanh Inverse Constraints:** The inverse hyperbolic tangent function has a limited domain, causing difficulties when the function's input exceeds this range.

## Conclusion

The hypothesis that outputs could be used to perfectly reconstruct inputs using inverses of activation functions and pseudo-inverses was not successful. However, the project demonstrates the potential of using autoencoders to approximate this process and highlights the complexities involved in such reconstructions.

## Running the Code

### Train the Autoencoder Model
```bash
python train.py
```

### Test the Trained Autoencoder Model
```bash
python test.py
```

### Test the Inverse Model
```bash
python inverse_test.py
```

## Configuration File

### config.yaml
```yaml
device: cuda
train:
  batch_size: 1024
  train_epochs: 90
  loss:
    - bceloss
    - regularizationloss
  optim:
    name: adam
    learning_rate: 0.001
    others:
  lr_scheduler:
    name: steplr
    others:
      step_size: 50
      gamma: 0.1
  alpha: 0.1
  model_path: ./pretrained
  progress_path: ./train_progress
  plot_epochs: 10
model:
  architecture:
    encoder:
        conv:
            - Linear:
                args: [784, 512]
                bias: False
            - Tanh:
            - Linear:
                args: [512, 256]
                bias: False
            - Tanh:
            - Flatten:
        mean:
            - Linear:
                args: [256, 32]
                bias: False
        log_var:
            - Linear:
                args: [256, 32]
                bias: False
    decoder:
        - Linear:
            args: [32, 256]
            bias: False
        - Tanh:
        - Linear:
            args: [256, 512]
            bias: False
        - Tanh:
        - Linear:
            args: [512, 784]
            bias: False
        - Sigmoid:
test:
  batch_size: 256
  n_components: 2
  model_path: ./pretrained/model_90.pt
  results_path: ./results
```

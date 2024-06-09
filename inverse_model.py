import torch
import torch.nn as nn

class InverseDecoderModel(nn.Module):
    def __init__(self, encoder_params):
        """
        Inverse Decoder Model.

        This model constructs an inverse of an encoder by reversing the layers and applying inverse activations.

        Args:
            encoder_params (dict): Dictionary containing the parameters of the encoder.
        """
        super(InverseDecoderModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(784, 256, bias=False),
            LeakyReLUInverse(),
            torch.nn.Linear(256, 32, bias=False),
        )

        for name, param in self.named_parameters():
            if 'layers.0.weight' in name:
                numpy_weight = encoder_params['encoder.1.weight']
                param.data = torch.tensor(numpy_weight).T
            elif 'layers.2.weight' in name:
                numpy_weight = encoder_params['encoder.3.weight']
                param.data = torch.tensor(numpy_weight).T

    def forward(self, x):
        """
        Forward pass of the Inverse Decoder Model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the inverse decoding process.
        """
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x
    
class TanhInverse(nn.Module):
    def __init__(self):
        """
        Tanh Inverse Activation Layer.

        This layer computes the inverse of the hyperbolic tangent function.
        """
        super(TanhInverse, self).__init__()

    def forward(self, x):
        """
        Forward pass of the Tanh Inverse Layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the inverse hyperbolic tangent function.
        """
        x = torch.clamp(x, -0.999, 0.999)
        return torch.atanh(x)

class SigmoidInverse(nn.Module):
    def __init__(self):
        """
        Sigmoid Inverse Activation Layer.

        This layer computes the inverse of the sigmoid function.
        """
        super(SigmoidInverse, self).__init__()

    def forward(self, x):
        """
        Forward pass of the Sigmoid Inverse Layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the inverse sigmoid function.
        """
        x = torch.clamp(x, 0.001, 0.999)
        return torch.logit(x)


class LeakyReLUInverse(nn.Module):
    def __init__(self, negative_slope=0.01):
        """
        LeakyReLU Inverse Activation Layer.

        This layer computes the inverse of the leaky relu function.
        """
        super(LeakyReLUInverse, self).__init__()
        self.inv_negative_slope = 1 / negative_slope

    def forward(self, x):
        """
        Forward pass of the LeakyReLU Inverse Layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the inverse leaky relu function.
        """
        output = (x >= 0) * x + (x < 0) * (x * self.inv_negative_slope)
        return output

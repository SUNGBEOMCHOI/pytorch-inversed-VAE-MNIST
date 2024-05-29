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
        self.layers = nn.ModuleList()
        self.layers.append(SigmoidInverse())  # Add Sigmoid inverse at the beginning

        for name, param in encoder_params.items():
            if 'weight' in name:
                out_features, in_features = param.shape
                layer = nn.Linear(in_features, out_features, bias=False)
                layer.weight = nn.Parameter(torch.tensor(param))
                self.layers.append(layer)
                self.layers.append(TanhInverse())  # Add Tanh inverse after each weight layer
            elif 'bias' in name:
                self.layers[-2].bias = nn.Parameter(torch.tensor(param))  # Attach bias to the corresponding layer

        # Remove the last TanhInverse added
        self.layers = self.layers[:-1]

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

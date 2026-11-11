"""Define the PyTorch neural network for bike rental prediction."""

import torch
from torch import nn


class BikeDemandNet(nn.Module):
    """A feedforward neural network with two or more hidden layers."""

    def __init__(self, input_size, hidden_sizes=(64, 32)):
        super().__init__()

        # Require at least two hidden layers.
        if len(hidden_sizes) < 2:
            raise ValueError("The network requires at least two hidden layers.")

        # Build the hidden layers using Linear and ReLU.
        layers = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.ReLU())
            previous_size = hidden_size

        # The output has one value with no activation function.
        layers.append(nn.Linear(previous_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features):
        """Predict the number of bike rentals for each input row."""
        return self.network(features)


# Verify the network when this file is run directly.
if __name__ == "__main__":
    model = BikeDemandNet(input_size=8)

    example_batch = torch.randn(64, 8)
    predictions = model(example_batch)

    print(model)
    print("Input shape:", example_batch.shape)
    print("Output shape:", predictions.shape)

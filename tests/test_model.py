"""Test the architecture and weight updates of the bike demand network."""

import pytest
import torch
from torch import nn

from model import BikeDemandNet


# Test several hidden-layer configurations using pytest parametrization.
@pytest.mark.parametrize(
    "hidden_sizes",
    [
        (16, 8),
        (32, 16),
        (64, 32),
        (128, 64, 32),
    ],
)
def test_network_output_shape_for_hidden_sizes(hidden_sizes):
    """The network must return one value per input row."""
    model = BikeDemandNet(input_size=8, hidden_sizes=hidden_sizes)
    features = torch.randn(10, 8)

    predictions = model(features)

    assert predictions.shape == (10, 1)
    assert predictions.dtype == torch.float32


# Confirm the network requires at least two hidden layers.
def test_network_rejects_fewer_than_two_hidden_layers():
    """Reject architectures with fewer than two hidden layers."""
    with pytest.raises(ValueError, match="at least two hidden layers"):
        BikeDemandNet(input_size=8, hidden_sizes=(32,))


# Confirm all hidden layers have ReLU and output has no activation.
def test_network_architecture():
    """Check the required hidden layers, ReLU, and single linear output."""
    model = BikeDemandNet(input_size=8, hidden_sizes=(64, 32))
    layers = list(model.network)

    assert isinstance(layers[0], nn.Linear)
    assert isinstance(layers[1], nn.ReLU)
    assert isinstance(layers[2], nn.Linear)
    assert isinstance(layers[3], nn.ReLU)
    assert isinstance(layers[4], nn.Linear)

    assert layers[-1].out_features == 1


# Check that one optimizer step updates the weights in every linear layer.
def test_optimizer_step_changes_every_layer():
    """Verify backpropagation and optimization update every layer."""
    torch.manual_seed(42)

    model = BikeDemandNet(input_size=8, hidden_sizes=(64, 32))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    features = torch.randn(32, 8)
    targets = torch.randn(32, 1)

    original_weights = [
        layer.weight.detach().clone()
        for layer in model.network
        if isinstance(layer, nn.Linear)
    ]

    predictions = model(features)
    loss = loss_fn(predictions, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    updated_weights = [
        layer.weight.detach().clone()
        for layer in model.network
        if isinstance(layer, nn.Linear)
    ]

    assert len(original_weights) == len(updated_weights) == 3

    for before, after in zip(original_weights, updated_weights, strict=True):
        assert not torch.equal(before, after)

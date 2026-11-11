"""Tests for the custom PyTorch training and validation loop."""

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import BikeDemandNet
from train import run_epoch, train_model


# Create a small, repeatable dataset for testing the training loop.
@pytest.fixture
def small_loader():
    """Return a small batch with predictable targets."""
    torch.manual_seed(42)

    features = torch.randn(16, 8)
    targets = (3 * features[:, 0] - 2 * features[:, 1] + 5).unsqueeze(1)

    return DataLoader(
        TensorDataset(features, targets),
        batch_size=16,
        shuffle=False,
    )


# Verify the training loop can memorize a small batch.
def test_network_memorizes_small_batch(small_loader):
    """Loss should fall below 5% of its starting value."""
    torch.manual_seed(42)

    model = BikeDemandNet(input_size=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    initial_loss, _ = run_epoch(model, small_loader, loss_fn)

    for _ in range(500):
        run_epoch(model, small_loader, loss_fn, optimizer)

    final_loss, _ = run_epoch(model, small_loader, loss_fn)

    assert final_loss < initial_loss * 0.05


# Confirm evaluation does not update any network weights.
def test_validation_does_not_change_weights(small_loader):
    """Validation must leave every model parameter unchanged."""
    torch.manual_seed(42)

    model = BikeDemandNet(input_size=8)
    loss_fn = nn.MSELoss()

    original_state = copy.deepcopy(model.state_dict())

    run_epoch(model, small_loader, loss_fn)

    for name, original_value in original_state.items():
        assert torch.equal(model.state_dict()[name], original_value)


# Check the history contains one row per epoch and all required metrics.
def test_training_history_has_required_fields(small_loader):
    """Training history must contain the required metrics for every epoch."""
    torch.manual_seed(42)

    model = BikeDemandNet(input_size=8)

    _, history, best_epoch = train_model(
        model,
        small_loader,
        small_loader,
        epochs=5,
        learning_rate=0.01,
    )

    assert len(history) == 5
    assert 1 <= best_epoch <= 5

    required_keys = {
        "epoch",
        "train_loss",
        "train_mae",
        "val_loss",
        "val_mae",
    }

    for row in history:
        assert required_keys.issubset(row.keys())

    assert [row["epoch"] for row in history] == list(range(1, 6))


# Check that training restores the best validation-loss checkpoint.
def test_training_restores_best_validation_weights(small_loader):
    """Restored model loss must match the lowest recorded validation loss."""
    torch.manual_seed(42)

    model = BikeDemandNet(input_size=8)

    model, history, best_epoch = train_model(
        model,
        small_loader,
        small_loader,
        epochs=15,
        learning_rate=0.01,
    )

    loss_fn = nn.MSELoss()
    restored_loss, _ = run_epoch(model, small_loader, loss_fn)

    best_recorded_loss = min(row["val_loss"] for row in history)

    assert best_epoch == min(history, key=lambda row: row["val_loss"])["epoch"]

    assert restored_loss == pytest.approx(
        best_recorded_loss,
        rel=1e-5,
        abs=1e-5,
    )


# Confirm invalid prediction and target shapes are rejected.
def test_run_epoch_rejects_mismatched_shapes():
    """Prevent accidental broadcasting when calculating training loss."""
    torch.manual_seed(42)

    model = BikeDemandNet(input_size=8)
    features = torch.randn(8, 8)

    # Incorrect target shape: (8,) instead of (8, 1).
    targets = torch.randn(8)
    loader = DataLoader(TensorDataset(features, targets), batch_size=8)

    with pytest.raises(ValueError, match="shapes must match"):
        run_epoch(model, loader, nn.MSELoss())


# Confirm that the training function rejects invalid epoch counts.
def test_training_rejects_zero_epochs(small_loader):
    """A training run must contain at least one epoch."""
    model = BikeDemandNet(input_size=8)

    with pytest.raises(ValueError, match="at least 1"):
        train_model(model, small_loader, small_loader, epochs=0)


# Confirm that the training function rejects invalid learning rates.
def test_training_rejects_invalid_learning_rate(small_loader):
    """The learning rate must be positive."""
    model = BikeDemandNet(input_size=8)

    with pytest.raises(ValueError, match="positive"):
        train_model(
            model,
            small_loader,
            small_loader,
            epochs=2,
            learning_rate=0,
        )


# Verify that training restores an earlier checkpoint, not the final weights.
def test_restores_best_epoch_when_last_epoch_is_worse(monkeypatch, small_loader):
    """Restore the best weights when validation worsens in later epochs."""
    torch.manual_seed(42)

    model = BikeDemandNet(input_size=8)
    initial_state = copy.deepcopy(model.state_dict())

    # Simulate three training epochs and their validation losses.
    validation_losses = [5.0, 2.0, 8.0]
    training_states = []
    call_count = 0

    # Replace run_epoch temporarily to control the validation results.
    def fake_run_epoch(model, loader, loss_fn, optimizer=None):
        nonlocal call_count

        epoch_index = call_count // 2
        is_training = optimizer is not None

        if is_training:
            # Simulate a different set of weights after each training epoch.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(1.0)

            training_states.append(copy.deepcopy(model.state_dict()))
            result = (10.0 - epoch_index, 3.0)
        else:
            result = (validation_losses[epoch_index], 2.0)

        call_count += 1
        return result

    monkeypatch.setattr("train.run_epoch", fake_run_epoch)

    trained_model, history, best_epoch = train_model(
        model,
        small_loader,
        small_loader,
        epochs=3,
        learning_rate=0.01,
    )

    # Epoch 2 has the lowest validation loss.
    assert best_epoch == 2
    assert len(history) == 3

    # The returned weights must match epoch 2, not epoch 3.
    for name, expected in training_states[1].items():
        assert torch.equal(trained_model.state_dict()[name], expected)

    # Confirm the final epoch had different weights.
    assert any(
        not torch.equal(trained_model.state_dict()[name], final_value)
        for name, final_value in training_states[2].items()
    )

    # Confirm training actually changed the initial weights.
    assert any(
        not torch.equal(trained_model.state_dict()[name], original_value)
        for name, original_value in initial_state.items()
    )

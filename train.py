"""Train bike-demand networks, compare baselines, and save reproducible reports."""

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from experiments import EXPERIMENTS
from model import BikeDemandNet
from prepare import make_loaders, prepare_data, split_data

SEED = 42
REPORTS_DIR = Path("reports")


# Train for one epoch when an optimizer is supplied; otherwise only evaluate.
def run_epoch(model, loader, loss_fn, optimizer=None):
    """Return sample-weighted mean loss and MAE in rentals."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_absolute_error = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for features, targets in loader:
            predictions = model(features)
            if predictions.shape != targets.shape:
                raise ValueError(
                    "Prediction and target shapes must match: "
                    f"{predictions.shape} != {targets.shape}"
                )

            loss = loss_fn(predictions, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_absolute_error += (
                torch.abs(predictions.detach() - targets).sum().item()
            )
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The DataLoader must contain at least one sample.")

    return total_loss / total_samples, total_absolute_error / total_samples


# Train for many epochs and restore the weights with the lowest validation loss.
def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001):
    """Return the best-checkpoint model, epoch history, and best epoch."""
    if epochs < 1:
        raise ValueError("The number of epochs must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("The learning rate must be positive.")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_weights = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )

        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}"
        )

    model.load_state_dict(best_weights)
    return model, history, best_epoch


# Compare the training-mean and linear regression baselines on the same splits.
def evaluate_baselines(train_data, val_data, test_data=None):
    """Report validation MAE and optional final test MAE for both baselines."""
    x_train, y_train = train_data
    x_val, y_val = val_data

    training_mean = float(np.mean(y_train))
    linear_model = LinearRegression()
    linear_model.fit(x_train, y_train)

    results = {
        "mean_baseline_val_mae": float(
            mean_absolute_error(y_val, np.full(len(y_val), training_mean))
        ),
        "linear_regression_val_mae": float(
            mean_absolute_error(y_val, linear_model.predict(x_val))
        ),
    }

    # Never pass test_data until the final network has been chosen.
    if test_data is not None:
        x_test, y_test = test_data
        results["mean_baseline_test_mae"] = float(
            mean_absolute_error(y_test, np.full(len(y_test), training_mean))
        )
        results["linear_regression_test_mae"] = float(
            mean_absolute_error(y_test, linear_model.predict(x_test))
        )

    return results


# Save learning curves for a run and mark its best validation-loss epoch.
def save_learning_curves(history, best_epoch, output_path, title):
    """Save training and validation MSE curves to a PNG file."""
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Training loss")
    ax.plot(epochs, val_losses, label="Validation loss")
    ax.axvline(best_epoch, color="red", linestyle="--", label=f"Best: {best_epoch}")
    ax.set(xlabel="Epoch", ylabel="Mean squared error (MSE)", title=title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Visualize the nonlinear bike-demand pattern before interpreting the baselines.
def save_hour_plot(frame, output_path):
    """Save average bike rentals by hour to support the README discussion."""
    hourly_mean = frame.groupby("hour")["rentals"].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hourly_mean.index, hourly_mean.values, marker="o")
    ax.set(
        xlabel="Hour of day",
        ylabel="Average rentals",
        title="Average bike rentals by hour",
    )
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Run an independent experiment without looking at the held-out test labels.
def run_experiment(name, config, train_data, val_data, test_data, reports_dir):
    """Train a fresh network and save its training curve and validation results."""
    torch.manual_seed(SEED)
    train_loader, val_loader, _ = make_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=config["batch_size"],
        seed=SEED,
    )
    model = BikeDemandNet(
        input_size=train_data[0].shape[1],
        hidden_sizes=config["hidden_sizes"],
    )
    model, history, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        epochs=config["epochs"],
        learning_rate=config["learning_rate"],
    )

    curve_filename = "learning_curves.png" if name == "main" else f"{name}_curves.png"
    save_learning_curves(
        history,
        best_epoch,
        reports_dir / curve_filename,
        title=f"Learning curves - {name}",
    )

    best_row = history[best_epoch - 1]
    final_row = history[-1]
    result = {
        "experiment": name,
        "hidden_sizes": str(config["hidden_sizes"]),
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "epochs": config["epochs"],
        "best_epoch": best_epoch,
        "best_val_loss": best_row["val_loss"],
        "best_val_mae": best_row["val_mae"],
        "final_train_loss": final_row["train_loss"],
        "final_val_loss": final_row["val_loss"],
    }
    return model, history, result


# Rebuild every report from the same split, training seeds, and experiments.
def main():
    """Run all experiments, choose a model, then evaluate test rows once."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    train, validation, test = split_data(seed=SEED)
    train_data, val_data, test_data, _ = prepare_data(train, validation, test)
    save_hour_plot(train, REPORTS_DIR / "rentals_by_hour.png")

    # Model comparison and selection use validation data only.
    baseline_results = evaluate_baselines(train_data, val_data)
    experiment_results = []
    trained_models = {}

    for name, config in EXPERIMENTS.items():
        print(f"\n{'=' * 60}\nRunning experiment: {name}\n{'=' * 60}")
        model, history, result = run_experiment(
            name, config, train_data, val_data, test_data, REPORTS_DIR
        )
        experiment_results.append(result)
        trained_models[name] = model

        if name == "main":
            pd.DataFrame(history).to_csv(REPORTS_DIR / "history.csv", index=False)

    pd.DataFrame(experiment_results).to_csv(
        REPORTS_DIR / "experiments.csv", index=False
    )

    # Within each run, the best checkpoint minimizes validation loss.
    # Across runs, select by the MAE of those restored checkpoints.
    selected_result = min(experiment_results, key=lambda row: row["best_val_mae"])
    selected_name = selected_result["experiment"]
    selected_model = trained_models[selected_name]
    print(f"\nSelected model (validation only): {selected_name}")

    # The test set is only evaluated after the final model is selected.
    _, _, test_loader = make_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=EXPERIMENTS[selected_name]["batch_size"],
        seed=SEED,
    )
    test_loss, test_mae = run_epoch(selected_model, test_loader, torch.nn.MSELoss())
    final_baselines = evaluate_baselines(train_data, val_data, test_data=test_data)

    metrics = {
        **baseline_results,
        "mean_baseline_test_mae": final_baselines["mean_baseline_test_mae"],
        "linear_regression_test_mae": final_baselines["linear_regression_test_mae"],
        "selected_model": selected_name,
        "network_val_mae": selected_result["best_val_mae"],
        "network_test_mae": test_mae,
        "network_test_loss": test_loss,
        "seed": SEED,
    }
    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=4))
    print("\nReports saved successfully.")


if __name__ == "__main__":
    main()

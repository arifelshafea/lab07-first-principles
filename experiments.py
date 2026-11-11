"""Configuration for the required neural network experiments."""

# Each experiment changes one setting relative to the main model.
EXPERIMENTS = {
    "main": {
        "hidden_sizes": (64, 32),
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 64,
    },
    "small_learning_rate": {
        "hidden_sizes": (64, 32),
        "learning_rate": 0.0001,
        "epochs": 100,
        "batch_size": 64,
    },
    "wide_network": {
        "hidden_sizes": (128, 64),
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 64,
    },
}

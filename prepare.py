"""Prepare bike rental data for training, validation, and testing."""

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

# Fixed seed makes data splitting and training batches reproducible.
RANDOM_STATE = 42

# Input features available when predicting bike rentals.
NUMERIC_FEATURES = [
    "hour",
    "temperature_c",
    "rain_mm",
    "wind_kph",
    "is_weekend",
]
CATEGORICAL_FEATURES = ["station_type"]


# Split the original data before fitting any preprocessing tools.
def split_data(seed=RANDOM_STATE):
    """Split 6,000 records into 60% train, 20% validation, and 20% test."""
    frame = make_rentals()

    # Reserve 20% for the final test set.
    train_val, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=seed,
        shuffle=True,
    )

    # Split the remaining 80% into 60% training and 20% validation.
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        random_state=seed,
        shuffle=True,
    )

    return train, validation, test


# Build a preprocessing tool for numeric and categorical features.
def build_preprocessor():
    """Scale numeric columns and one-hot encode station type."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        sparse_threshold=0.0,
    )


# Prepare the three splits without leaking validation or test information.
def prepare_data(train, validation, test):
    """Fit preprocessing on training data and transform all three splits."""
    preprocessor = build_preprocessor()

    x_train = preprocessor.fit_transform(train)
    x_val = preprocessor.transform(validation)
    x_test = preprocessor.transform(test)

    y_train = train[TARGET].to_numpy(dtype=np.float32)
    y_val = validation[TARGET].to_numpy(dtype=np.float32)
    y_test = test[TARGET].to_numpy(dtype=np.float32)

    return (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test),
        preprocessor,
    )


# Define our own Dataset class, returning one sample at a time.
class BikeRentalsDataset(Dataset):
    """A PyTorch Dataset of features and bike rental targets."""

    def __init__(self, features, targets):
        self.features = torch.as_tensor(np.asarray(features, dtype=np.float32))
        self.targets = torch.as_tensor(
            np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        )

        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets must have the same length.")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.targets)

    def __getitem__(self, index):
        """Return one feature row and one target as float32 tensors."""
        return self.features[index], self.targets[index]


# Create training, validation, and test DataLoaders.
def make_loaders(train_data, val_data, test_data, batch_size=64, seed=RANDOM_STATE):
    """Create DataLoaders, shuffling only the training data."""
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        BikeRentalsDataset(*train_data),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        BikeRentalsDataset(*val_data),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        BikeRentalsDataset(*test_data),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


# Check data preparation when this file is run directly.
if __name__ == "__main__":
    train, validation, test = split_data()
    train_data, val_data, test_data, preprocessor = prepare_data(
        train, validation, test
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_data, val_data, test_data
    )

    features, targets = next(iter(train_loader))

    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Test records:", len(test))
    print("Number of input features:", features.shape[1])
    print("Batch features shape:", features.shape)
    print("Batch targets shape:", targets.shape)
    print("Features dtype:", features.dtype)
    print("Targets dtype:", targets.dtype)

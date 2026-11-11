"""Test data splitting, preprocessing, Dataset, and DataLoaders."""

import numpy as np
import torch

from prepare import BikeRentalsDataset, make_loaders


# Check that all splits have the required sizes and do not overlap.
def test_split_sizes_and_no_overlap(rental_splits):
    """Verify the 60/20/20 split covers all records exactly once."""
    train, validation, test = rental_splits

    assert len(train) == 3600
    assert len(validation) == 1200
    assert len(test) == 1200
    assert len(train) + len(validation) + len(test) == 6000

    train_ids = set(train.index)
    val_ids = set(validation.index)
    test_ids = set(test.index)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


# Check that splitting the same data with the same seed gives identical rows.
def test_split_is_repeatable(rental_splits):
    """Confirm that the fixed random seed makes splitting repeatable."""
    from prepare import split_data

    repeated_splits = split_data(seed=42)

    for original, repeated in zip(rental_splits, repeated_splits, strict=True):
        assert original.equals(repeated)


# Confirm that the Dataset returns float32 tensors of the expected shapes.
def test_dataset_length_shapes_and_dtypes(prepared_rentals):
    """Verify a Dataset returns one feature row and one rental target."""
    train_data, _, _, _ = prepared_rentals
    dataset = BikeRentalsDataset(*train_data)

    assert len(dataset) == 3600

    features, target = dataset[0]

    assert features.shape == (8,)
    assert target.shape == (1,)
    assert features.dtype == torch.float32
    assert target.dtype == torch.float32


# Check that mismatched features and targets are rejected.
def test_dataset_rejects_mismatched_lengths():
    """Raise an error when features and targets have different lengths."""
    features = np.zeros((5, 8), dtype=np.float32)
    targets = np.zeros(4, dtype=np.float32)

    try:
        BikeRentalsDataset(features, targets)
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("Expected ValueError for mismatched lengths")


# Verify that only training rows are used to fit numeric preprocessing.
def test_scaler_is_fitted_on_training_only(prepared_rentals):
    """Training scaled numeric means are zero, unlike validation means."""
    train_data, val_data, _, _ = prepared_rentals
    x_train, _ = train_data
    x_val, _ = val_data

    # The first five columns are standardized numeric features.
    train_means = x_train[:, :5].mean(axis=0)
    val_means = x_val[:, :5].mean(axis=0)

    assert np.allclose(train_means, 0.0, atol=1e-6)
    assert not np.allclose(val_means, 0.0, atol=1e-6)


# Check that training batches shuffle consistently with a fixed seed.
def test_training_loader_shuffle_is_repeatable(prepared_rentals):
    """Same seed gives identical shuffled training batches."""
    train_data, val_data, test_data, _ = prepared_rentals

    loader_a, _, _ = make_loaders(
        train_data, val_data, test_data, batch_size=32, seed=42
    )
    loader_b, _, _ = make_loaders(
        train_data, val_data, test_data, batch_size=32, seed=42
    )

    features_a, targets_a = next(iter(loader_a))
    features_b, targets_b = next(iter(loader_b))

    assert torch.equal(features_a, features_b)
    assert torch.equal(targets_a, targets_b)

    # Training should shuffle rather than preserve original row order.
    original_features = torch.tensor(train_data[0][:32], dtype=torch.float32)
    assert not torch.equal(features_a, original_features)


# Confirm validation and test loaders preserve their original row order.
def test_validation_and_test_loaders_do_not_shuffle(prepared_rentals):
    """Evaluation loaders must preserve the input ordering."""
    train_data, val_data, test_data, _ = prepared_rentals

    _, val_loader, test_loader = make_loaders(
        train_data, val_data, test_data, batch_size=32, seed=42
    )

    val_features, _ = next(iter(val_loader))
    test_features, _ = next(iter(test_loader))

    expected_val = torch.tensor(val_data[0][:32], dtype=torch.float32)
    expected_test = torch.tensor(test_data[0][:32], dtype=torch.float32)

    assert torch.equal(val_features, expected_val)
    assert torch.equal(test_features, expected_test)

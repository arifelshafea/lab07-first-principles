"""Shared pytest fixtures for the bike demand project."""

import pytest

from prepare import prepare_data, split_data


# Create one reproducible data split for tests that need actual rental data.
@pytest.fixture(scope="module")
def rental_splits():
    """Return training, validation, and test DataFrames."""
    return split_data(seed=42)


# Prepare features using the training rows only.
@pytest.fixture(scope="module")
def prepared_rentals(rental_splits):
    """Return transformed training, validation, test data, and preprocessor."""
    train, validation, test = rental_splits
    return prepare_data(train, validation, test)

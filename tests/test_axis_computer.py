"""Tests for axis computation."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calibration.axis_computer import (
    compute_axis_vector,
    compute_normalization_scale,
    compute_dprime,
    compute_validation_metrics,
)


def test_compute_axis_vector():
    """Test contrastive axis computation."""
    np.random.seed(42)
    dim = 100

    # Create clearly separated distributions
    positive_states = [np.random.randn(dim) + np.array([1.0] + [0] * (dim - 1)) for _ in range(20)]
    negative_states = [np.random.randn(dim) - np.array([1.0] + [0] * (dim - 1)) for _ in range(20)]

    axis = compute_axis_vector(positive_states, negative_states)

    # Should be unit length
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-6

    # Should mostly point in first dimension
    assert axis[0] > 0.3, f"Expected first component > 0.3, got {axis[0]}"


def test_normalization_scale():
    """Test IQR-based normalization."""
    # Known distribution
    values = list(np.random.randn(100))
    scale = compute_normalization_scale(values)

    # IQR of standard normal ~ 1.35
    assert 0.5 < scale < 3.0, f"Scale {scale} out of expected range"


def test_normalization_scale_minimum():
    """Test that scale has a minimum to avoid division by zero."""
    scale = compute_normalization_scale([0.0, 0.0, 0.0, 0.0])
    assert scale >= 0.1


def test_dprime():
    """Test d-prime computation."""
    pos = [1.0, 1.1, 0.9, 1.2, 0.8]
    neg = [0.0, 0.1, -0.1, 0.2, -0.2]
    d = compute_dprime(pos, neg)
    assert d > 2.0, f"Expected large d-prime, got {d}"


def test_dprime_identical():
    """Test d-prime with identical distributions."""
    vals = [0.5, 0.5, 0.5, 0.5]
    d = compute_dprime(vals, vals)
    assert abs(d) < 0.01


def test_validation_metrics():
    """Test validation metrics computation."""
    np.random.seed(42)
    dim = 50
    axis_vector = np.zeros(dim)
    axis_vector[0] = 1.0  # Simple axis along first dimension

    # Add small noise so pooled_std > 0 (needed for d-prime)
    pos_states = [np.array([1.0] + [0.0] * (dim - 1)) + np.random.randn(dim) * 0.1 for _ in range(5)]
    neg_states = [np.array([-1.0] + [0.0] * (dim - 1)) + np.random.randn(dim) * 0.1 for _ in range(5)]

    metrics = compute_validation_metrics(axis_vector, pos_states, neg_states)

    assert metrics["accuracy"] == 1.0
    assert metrics["separation"] > 0
    assert metrics["dprime"] > 2.0

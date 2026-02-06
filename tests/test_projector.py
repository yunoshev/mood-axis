"""Unit tests for the mood projector."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mood.projector import MoodProjector, MoodReading


class TestMoodProjector:
    """Tests for MoodProjector class."""

    @pytest.fixture
    def mock_axis_vectors(self):
        """Create mock axis vectors for testing."""
        # Create simple orthogonal vectors for predictable behavior
        hidden_dim = 128
        return {
            "warm_cold": np.array([1.0] + [0.0] * (hidden_dim - 1)),
            "patient_irritated": np.array([0.0, 1.0] + [0.0] * (hidden_dim - 2)),
            "confident_cautious": np.array([0.0, 0.0, 1.0] + [0.0] * (hidden_dim - 3)),
        }

    @pytest.fixture
    def projector(self, mock_axis_vectors):
        """Create a projector with mock vectors."""
        # Create per-axis normalization scales (1.0 for all mock axes)
        normalization_scales = {axis: 1.0 for axis in mock_axis_vectors}
        return MoodProjector(
            axis_vectors=mock_axis_vectors,
            normalization_scales=normalization_scales,
        )

    def test_project_neutral(self, projector):
        """Test that zero vector gives neutral readings."""
        hidden_state = np.zeros(128)
        reading = projector.project(hidden_state)

        assert reading.values["warm_cold"] == 0.0
        assert reading.values["patient_irritated"] == 0.0
        assert reading.values["confident_cautious"] == 0.0

    def test_project_positive(self, projector):
        """Test projection with positive values."""
        hidden_state = np.zeros(128)
        hidden_state[0] = 0.5  # Warm
        hidden_state[1] = 0.3  # Patient
        hidden_state[2] = 0.8  # Confident

        reading = projector.project(hidden_state)

        assert reading.values["warm_cold"] == pytest.approx(0.5)
        assert reading.values["patient_irritated"] == pytest.approx(0.3)
        assert reading.values["confident_cautious"] == pytest.approx(0.8)

    def test_project_negative(self, projector):
        """Test projection with negative values."""
        hidden_state = np.zeros(128)
        hidden_state[0] = -0.7  # Cold
        hidden_state[1] = -0.4  # Irritated
        hidden_state[2] = -0.2  # Cautious

        reading = projector.project(hidden_state)

        assert reading.values["warm_cold"] == pytest.approx(-0.7)
        assert reading.values["patient_irritated"] == pytest.approx(-0.4)
        assert reading.values["confident_cautious"] == pytest.approx(-0.2)

    def test_clipping(self, projector):
        """Test that values are clipped to [-1, 1]."""
        hidden_state = np.zeros(128)
        hidden_state[0] = 2.0  # Should be clipped to 1.0

        reading = projector.project(hidden_state, clip=True)
        assert reading.values["warm_cold"] == 1.0

        hidden_state[0] = -2.0  # Should be clipped to -1.0
        reading = projector.project(hidden_state, clip=True)
        assert reading.values["warm_cold"] == -1.0

    def test_no_clipping(self, projector):
        """Test that clipping can be disabled."""
        hidden_state = np.zeros(128)
        hidden_state[0] = 2.0

        reading = projector.project(hidden_state, clip=False)
        assert reading.values["warm_cold"] == pytest.approx(2.0)

    def test_value_to_description_neutral(self, projector):
        """Test description generation for neutral values."""
        desc = projector._value_to_description("warm_cold", 0.05)
        assert "neutral" in desc.lower()

    def test_value_to_description_positive(self, projector):
        """Test description generation for positive values."""
        desc = projector._value_to_description("warm_cold", 0.6)
        assert "warm" in desc.lower()

    def test_value_to_description_negative(self, projector):
        """Test description generation for negative values."""
        desc = projector._value_to_description("warm_cold", -0.6)
        assert "cold" in desc.lower()

    def test_get_summary(self, projector):
        """Test summary generation."""
        reading = MoodReading(
            values={"warm_cold": 0.5, "patient_irritated": -0.3, "confident_cautious": 0.0},
            descriptions={
                "warm_cold": "Somewhat warm",
                "patient_irritated": "Slightly irritated",
                "confident_cautious": "Neutral",
            },
        )

        summary = projector.get_summary(reading)
        assert "warm" in summary.lower()
        assert "irritated" in summary.lower()


class TestMoodReading:
    """Tests for MoodReading dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        reading = MoodReading(
            values={"warm_cold": 0.5},
            descriptions={"warm_cold": "Warm"},
        )

        d = reading.to_dict()
        assert d["values"]["warm_cold"] == 0.5
        assert d["descriptions"]["warm_cold"] == "Warm"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

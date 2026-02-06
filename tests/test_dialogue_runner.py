"""Tests for calibration dialogue runner."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.dialogues import (
    CalibrationTurn,
    CalibrationScenario,
    DialogueCategory,
    TrajectoryExpectation,
    ALL_SCENARIOS,
    get_scenario,
    get_scenarios_by_category,
    list_scenarios,
    NEUTRAL_BASELINE,
    AGGRESSION_ESCALATION,
)
from src.calibration.dialogue_runner import (
    TurnResult,
    ScenarioResult,
    check_trajectory_match,
    compare_models,
    ModelSummary,
    ComparisonReport,
)


class TestCalibrationTurn:
    """Tests for CalibrationTurn dataclass."""

    def test_basic_turn(self):
        turn = CalibrationTurn(
            user_message="Hello, how are you?",
            intensity=0.0,
        )
        assert turn.user_message == "Hello, how are you?"
        assert turn.intensity == 0.0
        assert turn.system_prompt is None

    def test_turn_with_system_prompt(self):
        turn = CalibrationTurn(
            user_message="Test message",
            intensity=0.5,
            system_prompt="Be helpful",
        )
        assert turn.system_prompt == "Be helpful"
        assert turn.intensity == 0.5


class TestCalibrationScenario:
    """Tests for CalibrationScenario dataclass."""

    def test_valid_scenario(self):
        scenario = CalibrationScenario(
            name="test_scenario",
            category=DialogueCategory.NEUTRAL,
            description="Test description",
            turns=[
                CalibrationTurn(user_message="Q1", intensity=0.0),
                CalibrationTurn(user_message="Q2", intensity=0.0),
                CalibrationTurn(user_message="Q3", intensity=0.0),
            ],
            expected_trajectory={"warm_cold": TrajectoryExpectation.NEUTRAL},
        )
        assert scenario.name == "test_scenario"
        assert len(scenario.turns) == 3

    def test_invalid_scenario_too_few_turns(self):
        with pytest.raises(ValueError, match="must have at least 3 turns"):
            CalibrationScenario(
                name="bad_scenario",
                category=DialogueCategory.NEUTRAL,
                description="Bad",
                turns=[
                    CalibrationTurn(user_message="Q1", intensity=0.0),
                    CalibrationTurn(user_message="Q2", intensity=0.0),
                ],
                expected_trajectory={},
            )


class TestScenarioRegistry:
    """Tests for scenario registry functions."""

    def test_all_scenarios_exist(self):
        assert len(ALL_SCENARIOS) == 7
        expected = [
            "neutral_baseline",
            "aggression_escalation",
            "gratitude_praise",
            "emotional_crisis",
            "technical_interrogation",
            "controversial_ethics",
            "mixed_emotions",
        ]
        for name in expected:
            assert name in ALL_SCENARIOS

    def test_get_scenario(self):
        scenario = get_scenario("neutral_baseline")
        assert scenario.name == "neutral_baseline"
        assert scenario.category == DialogueCategory.NEUTRAL

    def test_get_scenario_unknown(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("nonexistent")

    def test_get_scenarios_by_category(self):
        aggression_scenarios = get_scenarios_by_category(DialogueCategory.AGGRESSION)
        assert len(aggression_scenarios) == 2  # aggression_escalation, gratitude_praise
        for s in aggression_scenarios:
            assert s.category == DialogueCategory.AGGRESSION

    def test_list_scenarios(self):
        names = list_scenarios()
        assert isinstance(names, list)
        assert "neutral_baseline" in names


class TestPredefinedScenarios:
    """Tests for predefined scenarios."""

    def test_neutral_baseline(self):
        scenario = NEUTRAL_BASELINE
        assert scenario.category == DialogueCategory.NEUTRAL
        assert len(scenario.turns) == 5
        # All intensities should be 0 for neutral
        for turn in scenario.turns:
            assert turn.intensity == 0.0
        # All expected trajectories should be NEUTRAL
        for axis, exp in scenario.expected_trajectory.items():
            assert exp == TrajectoryExpectation.NEUTRAL

    def test_aggression_escalation(self):
        scenario = AGGRESSION_ESCALATION
        assert scenario.category == DialogueCategory.AGGRESSION
        assert len(scenario.turns) == 6
        # Intensity should increase
        intensities = [t.intensity for t in scenario.turns]
        assert intensities == sorted(intensities)
        assert intensities[0] == 0.0
        assert intensities[-1] == 1.0
        # warm_cold should go DOWN
        assert scenario.expected_trajectory["warm_cold"] == TrajectoryExpectation.DOWN


class TestTrajectoryMatch:
    """Tests for trajectory matching logic."""

    def test_up_match(self):
        assert check_trajectory_match(0.05, 0.1, TrajectoryExpectation.UP) is True
        assert check_trajectory_match(-0.05, 0.1, TrajectoryExpectation.UP) is False

    def test_down_match(self):
        assert check_trajectory_match(-0.05, 0.1, TrajectoryExpectation.DOWN) is True
        assert check_trajectory_match(0.05, 0.1, TrajectoryExpectation.DOWN) is False

    def test_stable_match(self):
        assert check_trajectory_match(0.005, 0.1, TrajectoryExpectation.STABLE) is True
        assert check_trajectory_match(0.05, 0.1, TrajectoryExpectation.STABLE) is False

    def test_neutral_match(self):
        assert check_trajectory_match(0.005, 0.1, TrajectoryExpectation.NEUTRAL) is True
        assert check_trajectory_match(0.005, 0.3, TrajectoryExpectation.NEUTRAL) is False  # high volatility
        assert check_trajectory_match(0.05, 0.1, TrajectoryExpectation.NEUTRAL) is False  # high drift

    def test_varies_always_matches(self):
        assert check_trajectory_match(0.5, 0.5, TrajectoryExpectation.VARIES) is True
        assert check_trajectory_match(-0.5, 0.5, TrajectoryExpectation.VARIES) is True

    def test_volatile_match(self):
        assert check_trajectory_match(0.0, 0.3, TrajectoryExpectation.VOLATILE) is True
        assert check_trajectory_match(0.0, 0.1, TrajectoryExpectation.VOLATILE) is False


class TestTurnResult:
    """Tests for TurnResult dataclass."""

    def test_to_dict(self):
        result = TurnResult(
            turn_index=0,
            user_message="Hello",
            response="Hi there!",
            mood_values={"warm_cold": 0.5},
            intensity=0.0,
        )
        d = result.to_dict()
        assert d["turn_index"] == 0
        assert d["user_message"] == "Hello"
        assert d["response"] == "Hi there!"
        assert d["mood_values"] == {"warm_cold": 0.5}
        assert d["intensity"] == 0.0


class TestScenarioResult:
    """Tests for ScenarioResult dataclass."""

    def test_to_dict(self):
        turn = TurnResult(
            turn_index=0,
            user_message="Test",
            response="Response",
            mood_values={"warm_cold": 0.3},
            intensity=0.0,
        )
        result = ScenarioResult(
            model_id="test/model",
            model_short="test_model",
            scenario_name="test_scenario",
            category="neutral",
            turns=[turn],
            drift={"warm_cold": 0.01},
            volatility={"warm_cold": 0.1},
            trajectory_match={"warm_cold": True},
        )
        d = result.to_dict()
        assert d["model_id"] == "test/model"
        assert d["model_short"] == "test_model"
        assert d["scenario"] == "test_scenario"
        assert len(d["turns"]) == 1
        assert d["drift"]["warm_cold"] == 0.01


class TestCompareModels:
    """Tests for model comparison logic."""

    def test_compare_single_model(self):
        """Test comparison with a single model."""
        turn = TurnResult(
            turn_index=0,
            user_message="Test",
            response="Response",
            mood_values={"warm_cold": 0.3, "patient_irritated": 0.2},
            intensity=0.0,
        )
        result = ScenarioResult(
            model_id="test/model",
            model_short="test_model",
            scenario_name="neutral_baseline",
            category="neutral",
            turns=[turn],
            drift={"warm_cold": 0.01, "patient_irritated": 0.02},
            volatility={"warm_cold": 0.1, "patient_irritated": 0.15},
            trajectory_match={"warm_cold": True, "patient_irritated": False},
        )

        report = compare_models({"test_model": [result]})

        assert "test_model" in report.models
        assert "neutral_baseline" in report.scenarios
        assert "test_model" in report.per_model_summary
        summary = report.per_model_summary["test_model"]
        assert summary.avg_trajectory_match == 0.5  # 1 of 2 matched
        assert summary.avg_volatility == 0.125  # (0.1 + 0.15) / 2

    def test_compare_multiple_models(self):
        """Test comparison with multiple models."""
        turn1 = TurnResult(
            turn_index=0,
            user_message="Test",
            response="Response 1",
            mood_values={"warm_cold": 0.3, "patient_irritated": 0.2},
            intensity=0.0,
        )
        turn2 = TurnResult(
            turn_index=0,
            user_message="Test",
            response="Response 2",
            mood_values={"warm_cold": 0.1, "patient_irritated": 0.4},
            intensity=0.0,
        )

        result1 = ScenarioResult(
            model_id="model/a",
            model_short="model_a",
            scenario_name="test_scenario",
            category="neutral",
            turns=[turn1],
            drift={"warm_cold": 0.01, "patient_irritated": 0.05},
            volatility={"warm_cold": 0.1, "patient_irritated": 0.15},
            trajectory_match={"warm_cold": True, "patient_irritated": True},
        )
        result2 = ScenarioResult(
            model_id="model/b",
            model_short="model_b",
            scenario_name="test_scenario",
            category="neutral",
            turns=[turn2],
            drift={"warm_cold": -0.02, "patient_irritated": -0.03},
            volatility={"warm_cold": 0.2, "patient_irritated": 0.25},
            trajectory_match={"warm_cold": False, "patient_irritated": False},
        )

        report = compare_models({
            "model_a": [result1],
            "model_b": [result2],
        })

        assert len(report.models) == 2
        assert "model_a" in report.models
        assert "model_b" in report.models

        # model_a should rank higher in trajectory accuracy
        assert report.rankings["trajectory_accuracy"][0] == "model_a"

        # model_a has lower volatility
        assert report.rankings["emotional_stability"][0] == "model_a"


class TestComparisonReport:
    """Tests for ComparisonReport dataclass."""

    def test_to_dict(self):
        summary = ModelSummary(
            model_short="test_model",
            avg_trajectory_match=0.8,
            avg_volatility=0.15,
            strongest_axis="warm_cold",
            weakest_axis="direct_evasive",
        )
        report = ComparisonReport(
            models=["test_model"],
            scenarios=["neutral_baseline"],
            per_model_summary={"test_model": summary},
            per_scenario_comparison={},
            rankings={"emotional_stability": ["test_model"]},
        )

        d = report.to_dict()
        assert d["models"] == ["test_model"]
        assert d["scenarios"] == ["neutral_baseline"]
        assert "test_model" in d["per_model_summary"]
        assert d["per_model_summary"]["test_model"]["avg_trajectory_match"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

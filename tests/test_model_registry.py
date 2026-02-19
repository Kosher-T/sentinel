import unittest
import tempfile
import sys
from pathlib import Path

# Project setup
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from services.model_registry import ModelRegistry, ModelStatus


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.test_dir.name) / "test_registry.db"
        self.registry = ModelRegistry(db_path=self.db_path)

    def tearDown(self):
        self.test_dir.cleanup()

    # ----- Registration -----

    def test_register_model(self):
        """Register a model and verify version auto-increment."""
        v1 = self.registry.register_model(
            model_path="/models/challenger/v1.pth",
            source="retrain",
            trigger_reason="Drift Threshold Exceeded",
            training_metrics={"Loss": 0.05, "Accuracy": 0.92},
        )
        self.assertEqual(v1, "v1")

        v2 = self.registry.register_model(
            model_path="/models/challenger/v2.pth",
            source="retrain",
            trigger_reason="Scheduled",
        )
        self.assertEqual(v2, "v2")

    def test_register_stores_all_fields(self):
        """All provided fields are persisted."""
        version = self.registry.register_model(
            model_path="/models/challenger/v1.pth",
            source="manual",
            trigger_reason="Manual Upload",
            training_metrics={"Loss": 0.01, "Accuracy": 0.99, "epochs": 10},
            parent_version="v0",
            notes="Baseline model",
        )
        detail = self.registry.get_version(version)
        self.assertIsNotNone(detail)
        self.assertEqual(detail["source"], "manual")
        self.assertEqual(detail["trigger_reason"], "Manual Upload")
        self.assertEqual(detail["parent_version"], "v0")
        self.assertEqual(detail["notes"], "Baseline model")
        self.assertEqual(detail["status"], ModelStatus.REGISTERED)
        # Training metrics should be parsed dict
        self.assertIsInstance(detail["training_metrics_json"], dict)
        self.assertAlmostEqual(detail["training_metrics_json"]["Accuracy"], 0.99)

    # ----- Validation -----

    def test_validation_pass(self):
        """Register → validate (pass) → status becomes 'validated'."""
        version = self.registry.register_model("/m/v.pth", source="retrain")
        self.registry.update_validation(version, passed=True, decay_metrics={"score": 2.1})

        detail = self.registry.get_version(version)
        self.assertEqual(detail["status"], ModelStatus.VALIDATED)
        self.assertIsInstance(detail["decay_metrics_json"], dict)
        self.assertAlmostEqual(detail["decay_metrics_json"]["score"], 2.1)

    def test_validation_fail(self):
        """Register → validate (fail) → status becomes 'rejected'."""
        version = self.registry.register_model("/m/v.pth", source="retrain")
        self.registry.update_validation(version, passed=False, decay_metrics={"score": 8.5})

        detail = self.registry.get_version(version)
        self.assertEqual(detail["status"], ModelStatus.REJECTED)

    # ----- Deployment -----

    def test_deployment_flow(self):
        """Register → validate → deploy → verify get_current_production."""
        version = self.registry.register_model("/m/v.pth", source="retrain",
                                                training_metrics={"Accuracy": 0.95})
        self.registry.update_validation(version, passed=True)
        self.registry.record_deployment(version)

        detail = self.registry.get_version(version)
        self.assertEqual(detail["status"], ModelStatus.DEPLOYED)
        self.assertIsNotNone(detail["deployment_timestamp"])

        current = self.registry.get_current_production()
        self.assertIsNotNone(current)
        self.assertEqual(current["version"], version)

    def test_latest_deployment_wins(self):
        """When multiple models are deployed, get_current_production returns the latest."""
        v1 = self.registry.register_model("/m/v1.pth", source="retrain")
        self.registry.update_validation(v1, passed=True)
        self.registry.record_deployment(v1)

        v2 = self.registry.register_model("/m/v2.pth", source="retrain")
        self.registry.update_validation(v2, passed=True)
        self.registry.record_deployment(v2)

        current = self.registry.get_current_production()
        self.assertEqual(current["version"], "v2")

    # ----- History -----

    def test_version_history(self):
        """Multiple registrations return in newest-first order."""
        for i in range(5):
            self.registry.register_model(f"/m/v{i}.pth", source="retrain")

        history = self.registry.get_history(limit=10)
        self.assertEqual(len(history), 5)
        self.assertEqual(history[0]["version"], "v5")  # Newest first
        self.assertEqual(history[-1]["version"], "v1")  # Oldest last

    def test_history_limit(self):
        """History respects limit parameter."""
        for i in range(10):
            self.registry.register_model(f"/m/v{i}.pth", source="retrain")

        history = self.registry.get_history(limit=3)
        self.assertEqual(len(history), 3)

    # ----- Metrics Trend -----

    def test_metrics_trend(self):
        """Versions with metrics appear in chronological order for charting."""
        self.registry.register_model("/m/v1.pth", source="retrain",
                                      training_metrics={"Loss": 0.5, "Accuracy": 0.80})
        self.registry.register_model("/m/v2.pth", source="retrain",
                                      training_metrics={"Loss": 0.3, "Accuracy": 0.88})
        self.registry.register_model("/m/v3.pth", source="retrain")  # No metrics

        trend = self.registry.get_metrics_trend()
        # Only v1 and v2 should appear (v3 has no metrics)
        self.assertEqual(len(trend), 2)
        self.assertEqual(trend[0]["version"], "v1")  # Chronological order
        self.assertEqual(trend[1]["version"], "v2")
        self.assertAlmostEqual(trend[0]["Loss"], 0.5)
        self.assertAlmostEqual(trend[1]["Accuracy"], 0.88)

    # ----- Counts -----

    def test_counts(self):
        """Version and deployment counts are accurate."""
        self.assertEqual(self.registry.get_version_count(), 0)
        self.assertEqual(self.registry.get_deployment_count(), 0)

        v1 = self.registry.register_model("/m/v1.pth", source="retrain")
        v2 = self.registry.register_model("/m/v2.pth", source="retrain")

        self.assertEqual(self.registry.get_version_count(), 2)

        self.registry.update_validation(v1, passed=True)
        self.registry.record_deployment(v1)

        self.assertEqual(self.registry.get_deployment_count(), 1)

    # ----- Edge Cases -----

    def test_get_nonexistent_version(self):
        """Querying a version that doesn't exist returns None."""
        self.assertIsNone(self.registry.get_version("v999"))

    def test_no_production_before_deployment(self):
        """get_current_production returns None when nothing is deployed."""
        self.registry.register_model("/m/v1.pth", source="retrain")
        self.assertIsNone(self.registry.get_current_production())

    def test_parent_version_tracking(self):
        """Parent version lineage is correctly recorded."""
        v1 = self.registry.register_model("/m/v1.pth", source="retrain")
        self.registry.update_validation(v1, passed=True)
        self.registry.record_deployment(v1)

        v2 = self.registry.register_model("/m/v2.pth", source="retrain",
                                           parent_version=v1)
        detail = self.registry.get_version(v2)
        self.assertEqual(detail["parent_version"], "v1")


if __name__ == '__main__':
    unittest.main()

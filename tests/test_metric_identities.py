import unittest

import numpy as np

from scripts.metrics.churn_metrics import decision_churn_rates, jensen_shannon


class MetricIdentityTests(unittest.TestCase):
    def test_churn_decomposition_and_hidden_identity(self):
        dense = np.array([0, 0, 1, 1, 2, 2])
        compressed = np.array([0, 1, 1, 2, 0, 2])
        gold = np.array([0, 0, 0, 2, 1, 2])
        rates = decision_churn_rates(dense, compressed, gold)

        decomposed = (
            rates["damage"]
            + rates["repair"]
            + rates["wrong_to_wrong"]
            + rates["correct_to_correct_flip"]
        )
        self.assertAlmostEqual(rates["pred_flip"], decomposed)
        self.assertAlmostEqual(
            rates["accuracy_change"], rates["repair"] - rates["damage"]
        )
        self.assertAlmostEqual(
            rates["hidden_churn"],
            rates["wrong_to_wrong"] + 2.0 * min(rates["damage"], rates["repair"]),
        )

    def test_js_is_symmetric_and_zero_on_identical_inputs(self):
        p = np.array([[0.8, 0.2], [0.4, 0.6]])
        q = np.array([[0.7, 0.3], [0.1, 0.9]])
        np.testing.assert_allclose(jensen_shannon(p, q), jensen_shannon(q, p))
        np.testing.assert_allclose(jensen_shannon(p, p), np.zeros(2), atol=1e-14)


if __name__ == "__main__":
    unittest.main()

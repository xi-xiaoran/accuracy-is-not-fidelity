import ast
import unittest
from pathlib import Path


class AccuracyMatchingPolicyTests(unittest.TestCase):
    def test_one_to_one_sort_is_outcome_blind(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "accuracy_matched"
            / "build_accuracy_matched_v1.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))

        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "greedy_one_to_one"
        )
        source = ast.get_source_segment(
            path.read_text(encoding="utf-8"), function
        )
        self.assertIsNotNone(source)
        self.assertIn('"abs_acc_change_gap_pp"', source)
        self.assertIn('"target_a"', source)
        self.assertIn('"target_b"', source)
        self.assertNotIn('"hidden_churn_gap_pp"', source)
        self.assertNotIn('"pred_flip_gap_pp"', source)


if __name__ == "__main__":
    unittest.main()

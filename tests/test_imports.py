import importlib
import unittest


class ImportTests(unittest.TestCase):
    def test_dependency_light_modules_import(self):
        modules = [
            "scripts.metrics.churn_metrics",
            "scripts.parsing.build_fullzoo_lowrank_pair_metrics_tables",
            "scripts.parsing.parse_lmeval_uncertainty_pair_general",
        ]
        for name in modules:
            with self.subTest(module=name):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()

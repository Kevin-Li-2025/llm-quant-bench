import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "summarize_energy.py"
SPEC = importlib.util.spec_from_file_location("summarize_energy", SCRIPT_PATH)
energy = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(energy)


class EnergyScriptTest(unittest.TestCase):
    def test_integrates_power_with_trapezoid_rule(self):
        samples = [
            (0.0, 100.0),
            (10.0, 200.0),
            (20.0, 100.0),
        ]
        self.assertEqual(energy.integrate_power_joules(samples), 3000.0)

    def test_reads_nvidia_smi_power_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "gpu_power.csv"
            path.write_text(
                "\n".join(
                    [
                        "2026-05-19T00:00:00Z,100.0, 50, 1000, 2000, 60",
                        "2026-05-19T00:00:10Z,200.0, 60, 1000, 2000, 61",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            samples = energy.read_power_samples(path)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0][1], 100.0)
        self.assertEqual(samples[1][1], 200.0)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import pathlib
import sys
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_experiment_readiness.py"
SPEC = importlib.util.spec_from_file_location("check_experiment_readiness", SCRIPT_PATH)
readiness = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["check_experiment_readiness"] = readiness
SPEC.loader.exec_module(readiness)


class ReadinessScriptTest(unittest.TestCase):
    def test_infers_runtime_requirements(self):
        requirements = readiness.infer_requirements(
            {
                "runtime": "llama.cpp",
                "serve_command": ["bash", "-lc", "llama-server -m /models/model.gguf"],
            }
        )
        self.assertIn("llama-server", requirements["commands"])
        self.assertIn("/models/model.gguf", requirements["paths"])

    def test_absolute_vllm_command_satisfies_vllm_requirement(self):
        requirements = readiness.infer_requirements(
            {
                "runtime": "vLLM",
                "requires": {"commands": ["/opt/env/bin/vllm"]},
                "serve_command": ["/opt/env/bin/vllm", "serve", "/models/model"],
            }
        )
        self.assertEqual(requirements["commands"], ["/opt/env/bin/vllm"])


if __name__ == "__main__":
    unittest.main()

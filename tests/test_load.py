import tempfile
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_quant_bench.client import ModelConfig
from llm_quant_bench.load import run_load_test, summarize_load_records


class LoadMockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        text = "hello world"
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            event = {"choices": [{"delta": {"content": text}}]}
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        payload = {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format, *args):
        return


class LoadSummaryTest(unittest.TestCase):
    def test_summary_reports_candidate_only_throughput(self):
        records = [
            {
                "ok": True,
                "latency_s": 2.0,
                "ttft_s": 0.5,
                "inter_token_latency_s": 0.1,
                "time_per_output_token_s": 0.1,
                "output_tokens": 20,
                "prompt_tokens": 10,
                "tokens_per_second": 10,
            },
            {
                "ok": False,
                "error": "timeout",
                "latency_s": 3.0,
            },
        ]
        summary = summarize_load_records(
            records,
            benchmark_duration_s=5.0,
            concurrency=2,
            model_name="candidate",
            stream=True,
        )
        self.assertEqual(summary["requests"]["total"], 2)
        self.assertEqual(summary["requests"]["successful"], 1)
        self.assertAlmostEqual(summary["requests"]["success_rate"], 0.5)
        self.assertAlmostEqual(summary["tokens"]["output_token_throughput"], 4.0)
        self.assertEqual(summary["errors"]["timeout"], 1)


class LoadEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LoadMockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def test_run_load_test_respects_request_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_load_test(
                model=ModelConfig(
                    name="mock-candidate",
                    base_url=self.base_url,
                    model="mock-model",
                ),
                prompts=["hello"],
                out_dir=Path(tmp),
                concurrency=4,
                stream=True,
                requests=3,
            )
            self.assertEqual(summary["requests"]["total"], 3)
            self.assertTrue((Path(tmp) / "load_results.jsonl").exists())
            self.assertTrue((Path(tmp) / "load_summary.json").exists())
            self.assertTrue((Path(tmp) / "load_report.md").exists())


if __name__ == "__main__":
    unittest.main()

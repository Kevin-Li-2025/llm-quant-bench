import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm_quant_bench.client import ModelConfig, OpenAIChatClient


class OpenAIMockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        if "stream_options" in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"stream_options should not be sent by default")
            return

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for token in ["hello", " ", "world"]:
                event = {"choices": [{"delta": {"content": token}}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            return

        payload = {
            "choices": [{"message": {"content": "hello world"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def log_message(self, format, *args):
        return


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIMockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)
        cls.server.server_close()

    def test_non_streaming_generation(self):
        client = OpenAIChatClient(
            ModelConfig(name="mock", base_url=self.base_url, model="mock-model")
        )
        result = client.generate("hi", stream=False)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.output_tokens, 2)
        self.assertIsNone(result.ttft_s)

    def test_streaming_generation_without_default_stream_options(self):
        client = OpenAIChatClient(
            ModelConfig(name="mock", base_url=self.base_url, model="mock-model")
        )
        result = client.generate("hi", stream=True)
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "hello world")
        self.assertGreater(result.output_tokens, 0)
        self.assertIsNotNone(result.ttft_s)


if __name__ == "__main__":
    unittest.main()

"""In-process mock of an OpenAI-compatible ``/chat/completions`` endpoint.

Pure standard library so the test suite runs on a fully airgapped host: the
server binds to 127.0.0.1 on an ephemeral port, so nothing ever leaves the
machine. Both the integration tests (which import ``benchmark`` and call it
directly) and the system tests (which spawn ``benchmark.py`` as a subprocess)
point at an instance of this server over loopback TCP.

Typical use::

    with MockServer(default_content="PONG") as server:
        content, elapsed, usage = benchmark.call_chat_api(
            server.base_url, "", "model", "hi", 16, 0, 5)

Swap in custom behavior with ``server.set_responder(fn)`` where ``fn`` is
``responder(request_json, server) -> (status_code, payload, delay_seconds)``.
``payload`` may be a dict/list (JSON-encoded), a str, or bytes.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def chat_body(content, completion_tokens=None):
    """Build a minimal OpenAI-style chat-completion response body."""
    body = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if completion_tokens is not None:
        body["usage"] = {
            "prompt_tokens": 1,
            "completion_tokens": completion_tokens,
            "total_tokens": completion_tokens + 1,
        }
    return body


def _default_responder(req, srv):
    return 200, chat_body(srv.default_content, srv.default_tokens), 0


def make_answer_responder(answers, default="i do not know", tokens=5):
    """Route by prompt content.

    ``answers`` is a list of ``(needle, content)`` pairs; the first ``needle``
    found as a substring of the request's prompt wins. Falls back to
    ``default`` when nothing matches.
    """
    def responder(req, srv):
        try:
            prompt = req["messages"][-1]["content"]
        except (KeyError, IndexError, TypeError):
            prompt = ""
        for needle, content in answers:
            if needle in prompt:
                return 200, chat_body(content, tokens), 0
        return 200, chat_body(default, tokens), 0
    return responder


def make_flaky_responder(fail_times, content="PONG", status=503, tokens=5):
    """Fail (HTTP ``status``) the first ``fail_times`` requests, then succeed."""
    state = {"n": 0}

    def responder(req, srv):
        if state["n"] < fail_times:
            state["n"] += 1
            return status, {"error": {"message": "temporary"}}, 0
        return 200, chat_body(content, tokens), 0
    return responder


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output clean
        pass

    def do_POST(self):
        srv = self.server
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            req = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            req = {}

        srv.requests.append({
            "path": self.path,
            "headers": {k: v for k, v in self.headers.items()},
            "json": req,
        })

        status, payload, delay = srv.responder(req, srv)
        if delay:
            time.sleep(delay)

        if isinstance(payload, (dict, list)):
            data = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = payload or b""

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class MockServer:
    """A threaded, loopback-only mock chat-completions server."""

    def __init__(self, responder=None, default_content="PONG", default_tokens=5):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.requests = []
        self.httpd.responder = responder or _default_responder
        self.httpd.default_content = default_content
        self.httpd.default_tokens = default_tokens
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def port(self):
        return self.httpd.server_address[1]

    @property
    def base_url(self):
        return "http://127.0.0.1:{}/v1".format(self.port)

    @property
    def requests(self):
        return self.httpd.requests

    def set_responder(self, fn):
        self.httpd.responder = fn

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

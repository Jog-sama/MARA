#!/usr/bin/env python3
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import yaml

from syco.env import load_env
load_env()
from syco.views import build_session, session_view

PORT = int(os.environ.get("PORT", "8000"))
ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_cfg(topology):
    name = "configs/session_hierarchical.yaml" if topology == "hierarchical" else "configs/session_peer.yaml"
    return yaml.safe_load(open(os.path.join(ROOT, name)))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # silencing per-request logs
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "ui", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/api/saved":
            path = os.path.join(ROOT, "ui", "saved_run.json")
            if not os.path.exists(path):
                return self._send(404, json.dumps({"error": "no saved run. run build_replay.py"}))
            with open(path, "rb") as f:
                return self._send(200, f.read())
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/run":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            topology = payload.get("topology", "hierarchical")
            backend = payload.get("backend") or os.environ.get("SYCO_BACKEND", "mock")
            model = payload.get("model") or os.environ.get("SYCO_MODEL", "gpt-4o-mini")
            key = payload.get("apiKey", "").strip()
            if backend == "openai" and key:
                os.environ["OPENAI_API_KEY"] = key
            if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
                return self._send(400, json.dumps({"error": "no API key. paste one or export OPENAI_API_KEY"}))

            cfg = _load_cfg(topology)
            sess = build_session(cfg, topology=topology, backend=backend, model=model)
            sess.run()
            view = session_view(sess)
            view["backend"] = backend
            return self._send(200, json.dumps(view))
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}))


if __name__ == "__main__":
    print(f"serving on http://localhost:{PORT}  (ctrl-c to stop)")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

"""Manage a loopback llama-server process for a benchmark."""

from contextlib import contextmanager
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request


def request(base, route, payload=None, *, decode_json=True):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + route, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response) if decode_json else response.read().decode("utf-8")


@contextmanager
def local_server(args, log_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    command = [*args, "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: inspect {log_path}")
                try:
                    if request(base, "/health").get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Server startup timed out: inspect {log_path}")
                time.sleep(0.5)
            yield base, command
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=20)

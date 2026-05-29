"""Quick Ollama connectivity test with configurable timeout."""
import sys
import requests

base = sys.argv[1] if len(sys.argv) > 1 else "http://10.17.18.142:11434"
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

print(f"base={base} timeout={timeout}s")

print("GET /api/tags")
try:
    r = requests.get(f"{base}/api/tags", timeout=timeout)
    print(f"  status {r.status_code}")
    print(f"  body {r.text[:200]}")
except Exception as e:
    print(f"  ERROR {type(e).__name__}: {e}")

print("POST /api/chat")
try:
    r = requests.post(
        f"{base}/api/chat",
        json={
            "model": "qwen3-vl:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "hi"}],
        },
        timeout=timeout,
    )
    print(f"  status {r.status_code}")
    print(f"  body {r.text[:200]}")
except Exception as e:
    print(f"  ERROR {type(e).__name__}: {e}")

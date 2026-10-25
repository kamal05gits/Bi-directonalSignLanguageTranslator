    #!/usr/bin/env python3
"""Signora — local launcher.

Starts the Flask app on http://localhost:5000 (override with PORT=xxxx)
and opens it in your default browser. Debug reloader is disabled so the
TensorFlow model is loaded only once.
"""
import os
import sys
import threading
import webbrowser

if sys.version_info[:2] != (3, 11):
    print(f"⚠️  Signora is tested on Python 3.11 — you are running {sys.version.split()[0]}.")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
for d in ("logs", "static/uploads", "data/raw/gestures"):
    os.makedirs(d, exist_ok=True)

# Browser webcam mode (same as the Hugging Face Space)
os.environ.setdefault("USE_SERVER_CAMERA", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from app import app  # noqa: E402  (loads the model)

port = int(os.environ.get("PORT", "5000"))
url = f"http://localhost:{port}"

if os.environ.get("NO_BROWSER") != "1":
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()

print("\n" + "=" * 56)
print(f"  🤟 Signora is running →  {url}")
print("  Press Ctrl+C to stop")
print("=" * 56 + "\n")
app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)

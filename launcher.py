"""
launcher.py — PyInstaller entry point for MindVault v2.

When the packaged .app / .exe is launched, this:
  1. Tries to start 'ollama serve' if it isn't already running.
  2. Finds a free port starting at 8501.
  3. Launches Streamlit (app.py) as a subprocess.
  4. Opens the browser after a short warm-up delay.

NOT intended for direct development use.
Development: streamlit run app.py
"""

import os
import sys
import time
import socket
import threading
import subprocess
import webbrowser
from pathlib import Path


def find_free_port(start: int = 8501) -> int:
    for p in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start


def ensure_ollama():
    """Best-effort: start ollama serve if not already responding."""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2)
        return  # Already up
    except Exception:
        pass

    if sys.platform in ("darwin", "linux"):
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(3)
        except FileNotFoundError:
            print("⚠️  Ollama not found. Install from https://ollama.com")


def open_browser(url: str, delay: float = 3.0):
    def _open():
        time.sleep(delay)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main():
    ensure_ollama()

    port    = find_free_port()
    url     = f"http://localhost:{port}"
    app_dir = Path(__file__).parent

    print(f"🔐 MindVault v2 → {url}")
    open_browser(url)

    # Disable Streamlit telemetry
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"]             = "true"

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_dir / "app.py"),
        "--server.port",     str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ])


if __name__ == "__main__":
    main()

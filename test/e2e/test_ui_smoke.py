import os
import time
import subprocess
import signal
import sys
import pathlib
import pytest

from playwright.sync_api import sync_playwright


E2E_ENABLED = os.getenv("E2E") == "1"


@pytest.mark.e2e
@pytest.mark.skipif(not E2E_ENABLED, reason="Set E2E=1 to run Playwright E2E test")
def test_ui_smoke(tmp_path):
    project_root = pathlib.Path(__file__).resolve().parents[2]
    backend_url = "http://localhost:8001"
    frontend_url = "http://localhost:5174"

    env = os.environ.copy()
    env["VITE_API_BASE_URL"] = backend_url

    popen_kwargs = {}
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = os.setsid

    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web_app.main:app", "--port", "8001"],
        cwd=str(project_root),
        env=env,
        text=True,
        **popen_kwargs,
    )

    try:
        wait_for_http(backend_url + "/", timeout=60)

        frontend = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", "5174"],
            cwd=str(project_root / "frontend"),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            **popen_kwargs,
        )

        try:
            wait_for_http(frontend_url + "/", timeout=120)

            artifacts = project_root / "test" / "e2e" / "artifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            screenshot_path = artifacts / "ui_smoke.png"

            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(frontend_url, wait_until="domcontentloaded")

                # Headline and lead text
                page.get_by_role("heading", name="PrivSyn").wait_for()
                page.get_by_text("A Tool for Differentially Private Data Synthesis").wait_for()

                page.screenshot(path=str(screenshot_path))
                browser.close()
        finally:
            terminate(frontend)
    finally:
        terminate(backend)


def wait_for_http(url: str, timeout: int = 60):
    import requests

    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return
        except Exception as e:
            last_err = e
        time.sleep(1)
    raise AssertionError(f"Timeout waiting for {url}. Last error: {last_err}")


def terminate(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        try:
            if sys.platform.startswith("win"):
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.kill()
        finally:
            proc.wait(timeout=30)

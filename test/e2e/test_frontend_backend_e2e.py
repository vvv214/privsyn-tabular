import os
import time
import subprocess
import signal
import sys
import pathlib
import requests
import pytest

from contextlib import contextmanager
from playwright.sync_api import sync_playwright


E2E_ENABLED = os.getenv("E2E") == "1"


@pytest.mark.e2e
@pytest.mark.skipif(not E2E_ENABLED, reason="Set E2E=1 to run Playwright E2E test")
def test_end_to_end_frontend_backend(tmp_path):
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

            sample_zip = project_root / "sample_data" / "adult.csv.zip"
            assert sample_zip.exists(), f"Sample file missing: {sample_zip}"

            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(frontend_url, wait_until="domcontentloaded")

                # Fill form
                page.fill("#dataset_name", "adult")
                page.set_input_files("#data_file", str(sample_zip))

                # Submit to infer metadata
                page.get_by_role("button", name="Infer Metadata & Synthesize").click()

                # Wait for Metadata confirmation view
                page.get_by_text("Confirm Inferred Metadata").wait_for(timeout=120_000)

                # Ensure categorical values are detected for at least one column
                gender_card = page.locator("div.card").filter(
                    has=page.get_by_role("heading", name="gender")
                )
                gender_select = gender_card.locator("select")
                gender_select.wait_for()
                options = gender_select.locator("option")
                assert options.count() > 0
                option_texts = options.all_text_contents()
                assert any("Female" in text or "Male" in text for text in option_texts)

                # Confirm & synthesize
                page.get_by_role("button", name="Confirm & Synthesize").click()

                # Wait for result page and preview
                page.get_by_text("Synthesis Results").wait_for(timeout=300_000)
                page.get_by_text("Synthesized Data Preview").wait_for(timeout=60_000)

                # Check download link is present
                assert page.get_by_role("link", name="Download Synthesized Data").is_visible()

                browser.close()
        finally:
            terminate(frontend)
    finally:
        terminate(backend)


def wait_for_http(url: str, timeout: int = 60):
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

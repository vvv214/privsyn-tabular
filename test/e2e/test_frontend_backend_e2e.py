import json
import os
import time
import subprocess
import signal
import sys
import pathlib
import requests
import pytest

from playwright.sync_api import sync_playwright, Error as PlaywrightError

E2E_ENABLED = os.getenv("E2E") == "1"


def run_e2e(callback):
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
                try:
                    bootstrap_metadata(page, frontend_url, sample_zip)
                    callback(page)
                finally:
                    page.close()
                    browser.close()
        finally:
            terminate(frontend)
    finally:
        terminate(backend)


def bootstrap_metadata(page, frontend_url, sample_zip):
    page.goto(frontend_url, wait_until="domcontentloaded")
    page.fill("#dataset_name", "adult")
    page.set_input_files("#data_file", str(sample_zip))
    page.get_by_role("button", name="Infer Metadata & Synthesize").click()
    page.get_by_text("Confirm Inferred Metadata").wait_for(timeout=120_000)


def assert_gender_options(page, retries=3):
    selector = "[data-testid='categorical-select-gender']"
    for attempt in range(retries):
        try:
            gender_values = page.locator(selector)
            gender_values.wait_for(state="visible")
            options = gender_values.locator("span")
            assert options.count() > 0, "Expected detected categorical values for gender"
            option_texts = options.all_text_contents()
            assert any(label.strip() for label in option_texts), "Detected values list should have non-empty labels"
            return
        except PlaywrightError:
            if attempt == retries - 1:
                raise
            page.wait_for_timeout(1_000)


def wait_for_results(page):
    page.get_by_text("Synthesis Results").wait_for(timeout=300_000)
    page.get_by_text("Synthesized Data Preview").wait_for(timeout=60_000)


@pytest.mark.e2e
@pytest.mark.skipif(not E2E_ENABLED, reason="Set E2E=1 to run Playwright E2E test")
def test_end_to_end_frontend_backend():
    def flow(page):
        assert_gender_options(page)
        page.get_by_role("button", name="Confirm & Synthesize").click()
        wait_for_results(page)
        assert page.get_by_role("link", name="Download Synthesized Data").is_visible()

    run_e2e(flow)


@pytest.mark.e2e
@pytest.mark.skipif(not E2E_ENABLED, reason="Set E2E=1 to run Playwright E2E test")
def test_metadata_validation_blocked():
    def flow(page):
        assert_gender_options(page)
        add_input = page.locator("[data-testid='category-input-gender']")
        add_input.fill("   ")
        page.get_by_test_id("category-add-gender").click()
        helper = page.get_by_test_id("category-helper-message")
        helper.wait_for()
        assert "Enter a non-empty value" in helper.inner_text()

    run_e2e(flow)


@pytest.mark.e2e
@pytest.mark.skipif(not E2E_ENABLED, reason="Set E2E=1 to run Playwright E2E test")
def test_numeric_bounds_validation_flow():
    def flow(page):
        assert_gender_options(page)
        page.locator("#domain_age_type").select_option("numerical")
        min_input = page.get_by_test_id("numeric-min-age")
        max_input = page.get_by_test_id("numeric-max-age")
        min_input.wait_for()
        max_input.wait_for()
        min_input.fill("90")
        max_input.fill("10")
        page.get_by_role("button", name="Confirm & Synthesize").click()
        alert = page.get_by_test_id("metadata-validation-alert")
        alert.wait_for()
        assert "Maximum must be greater than or equal to minimum" in alert.inner_text()

        # Fix the bounds and complete synthesis
        min_input.fill("0")
        max_input.fill("100")
        page.get_by_role("button", name="Confirm & Synthesize").click()
        wait_for_results(page)
        download_link = page.get_by_role("link", name="Download Synthesized Data")
        try:
            download_link.wait_for(state="visible", timeout=120_000)
        except PlaywrightError:
            warning = page.get_by_text("Evaluation warning")
            if warning.is_visible():
                retry_button = page.get_by_role("button", name="Retry evaluation")
                retry_button.click()
                page.get_by_text("Evaluation complete.").wait_for(timeout=60_000)
                download_link.wait_for(state="visible", timeout=120_000)
            else:
                raise
        href = download_link.get_attribute("href")
        assert href, "Download link missing href"
        session_id = href.rstrip("/").split("/")[-1]
        assert session_id, "Failed to parse session identifier"

    run_e2e(flow)


@pytest.mark.e2e
@pytest.mark.skipif(not E2E_ENABLED, reason="Set E2E=1 to run Playwright E2E test")
def test_evaluation_failure_with_retry():
    def flow(page):
        assert_gender_options(page)

        failure_flag = {"count": 0}

        def intercept(route, request):
            if failure_flag["count"] == 0:
                failure_flag["count"] += 1
                route.fulfill(
                    status=500,
                    headers={"Content-Type": "application/json"},
                    body=json.dumps({"detail": {"error": "simulated_failure", "message": "Simulated evaluation failure"}}),
                )
                page.unroute("**/evaluate", intercept)
            else:
                route.continue_()

        page.route("**/evaluate", intercept)
        page.get_by_role("button", name="Confirm & Synthesize").click()
        wait_for_results(page)

        # Even if evaluation fails, the download link should appear once retry succeeds.
        download_link = page.get_by_role("link", name="Download Synthesized Data")
        download_link.wait_for(state="visible", timeout=120_000)

    run_e2e(flow)


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

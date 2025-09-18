#!/usr/bin/env python3
"""Capture key UI screenshots using Playwright.

This script spins up the FastAPI backend plus the Vite dev server, walks
through the sample adult dataset flow, and saves screenshots under
``docs/media``. Use it whenever the UI changes instead of grabbing images
manually.

Requirements:
    * ``npm`` and ``python`` available on PATH
    * Playwright browsers installed (``playwright install``)
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Iterable, List

import requests
from playwright.sync_api import Playwright, sync_playwright


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MEDIA_DIR = PROJECT_ROOT / "docs" / "media"
SAMPLE_ZIP = PROJECT_ROOT / "sample_data" / "adult.csv.zip"

BACKEND_URL = "http://127.0.0.1:8001"
FRONTEND_URL = "http://127.0.0.1:5174"


def wait_for_http(url: str, timeout: int = 60) -> None:
    start = time.time()
    last_err: Exception | None = None
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001 - preserve last error for debugging
            last_err = exc
        time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {url}. Last error: {last_err}")


def is_port_open(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return False
        return True


def terminate(processes: Iterable[subprocess.Popen]) -> None:
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            if sys.platform == "win32":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    for proc in processes:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def bootstrap_metadata(page, sample_zip: pathlib.Path) -> None:
    page.goto(FRONTEND_URL, wait_until="domcontentloaded")
    page.fill("#dataset_name", "adult")
    page.set_input_files("#data_file", str(sample_zip))
    page.get_by_role("button", name="Infer Metadata & Synthesize").click()
    page.get_by_text("Confirm Inferred Metadata").wait_for(timeout=120_000)


def capture_screens(playwright: Playwright) -> None:
    browser = playwright.chromium.launch()
    page = browser.new_page()

    try:
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)

        # Landing form
        page.goto(FRONTEND_URL, wait_until="networkidle")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(MEDIA_DIR / "ui_form.png"), full_page=True)

        # Metadata confirmation
        bootstrap_metadata(page, SAMPLE_ZIP)
        page.wait_for_timeout(1000)

        metadata_card = page.locator("div.card", has_text="Dataset Information").first
        metadata_card.scroll_into_view_if_needed()
        metadata_card.screenshot(path=str(MEDIA_DIR / "ui_metadata_overview.png"))

        first_column_card = page.locator("div.card.h-100").first
        first_column_card.scroll_into_view_if_needed()
        first_column_card.screenshot(path=str(MEDIA_DIR / "ui_metadata_column.png"))

        # Results page
        page.get_by_role("button", name="Confirm & Synthesize").click()
        page.get_by_text("Synthesis Results").wait_for(timeout=300_000)
        page.wait_for_timeout(1000)
        results_card = page.locator("section.card").first
        results_card.scroll_into_view_if_needed()
        results_card.screenshot(path=str(MEDIA_DIR / "ui_results.png"))
    finally:
        page.close()
        browser.close()


def main() -> None:
    if not SAMPLE_ZIP.exists():
        raise FileNotFoundError(f"Sample dataset missing: {SAMPLE_ZIP}")

    env = os.environ.copy()
    env["VITE_API_BASE_URL"] = BACKEND_URL

    popen_kwargs: dict[str, object] = {}
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = os.setsid

    spawned: List[subprocess.Popen] = []

    backend_running = is_port_open(8001)
    if backend_running:
        print("Backend already running on port 8001. Skipping launch.")
    else:
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "web_app.main:app", "--port", "8001"],
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            **popen_kwargs,
        )
        spawned.append(backend)

    try:
        wait_for_http(f"{BACKEND_URL}/", timeout=90)

        frontend_running = is_port_open(5174)
        if frontend_running:
            print("Frontend already running on port 5174. Skipping launch.")
            frontend = None
        else:
            frontend = subprocess.Popen(
                ["npm", "run", "dev", "--", "--port", "5174"],
                cwd=str(FRONTEND_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                **popen_kwargs,
            )
            spawned.append(frontend)

        try:
            wait_for_http(FRONTEND_URL, timeout=120)

            with sync_playwright() as playwright:
                capture_screens(playwright)
        finally:
            if frontend is not None:
                terminate([frontend])
    finally:
        terminate(spawned)


if __name__ == "__main__":
    main()

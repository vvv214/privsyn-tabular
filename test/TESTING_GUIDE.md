# How to Run Tests

This guide explains how to run the Python tests for the PrivSyn project using `pytest`.

## Prerequisites

*   Python 3.x installed.
*   Project dependencies installed (preferably in a virtual environment).

## Steps to Run Tests

1.  **Navigate to the Project Root:**
    Open your terminal or command prompt and navigate to the root directory of the `PrivSyn` project.

    ```bash
    cd /path/to/your/PrivSyn/project
    ```
    (Replace `/path/to/your/PrivSyn/project` with the actual path on your system, e.g., `/Users/x/Documents/GitHub/PrivSyn`)

2.  **Activate Virtual Environment (Recommended):**
    If you are using a virtual environment (which is highly recommended for dependency management), activate it.

    ```bash
    source ./.venv/bin/activate
    ```
    (On Windows, it might be `.\.venv\Scripts\activate`)

3.  **Run Pytest:**
    You can run specific test files or all tests in the `test/` directory.

    *   **To run the preprocessing pipeline test:**
        ```bash
        ./.venv/bin/pytest test/test_preprocessing.py
        ```

    *   **To run the GUM synthesis pipeline test:**
        ```bash
        ./.venv/bin/pytest test/test_synthesis.py
        ```

    *   **To run all tests in the `test/` directory:**
        ```bash
        ./.venv/bin/pytest test/
        ```

## Interpreting Test Results

*   **`PASSED`**: Indicates that the test function completed successfully without any unhandled exceptions and all assertions passed.
*   **`FAILED`**: Indicates that an assertion failed or an unhandled exception occurred during the test execution. Pytest will show a detailed traceback.
*   **`ERROR`**: Indicates an error occurred during test setup or teardown, or if the test environment itself is problematic.
*   **`SKIPPED`**: Indicates that a test was intentionally skipped (e.g., due to missing dependencies or specific conditions).

If a test `FAILED`, examine the traceback provided by `pytest` to identify the line of code causing the issue. The output will also include any `print()` statements from your test code.

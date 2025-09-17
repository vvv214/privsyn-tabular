# Frontend Guide

## High-Level Flow

1. **Upload Form (`SynthesisForm.jsx`)**
   - Validates file type (`.csv`/`.zip`), ensures consistent row lengths, and displays inline errors.
   - Supports loading the sample dataset without a file.
2. **Metadata Confirmation (`MetadataConfirmation.jsx`)**
   - Normalises inferred domain data so categorical/numerical editors always have sensible defaults.
   - Validation rules (as of now):
     - Categorical columns must retain at least one category (selected or custom).
     - Numerical columns require both min and max and enforce `max ≥ min`.
   - Shows aggregated errors before allowing submission.
3. **Results (`ResultsDisplay.jsx`)**
   - Displays download link, preview table, and evaluation JSON payloads.

These steps mirror the Playwright E2E scenario in `test/e2e/test_frontend_backend_e2e.py`. When evolving the UI, align component changes with assertions there to keep regressions visible.

## Components at a Glance

| Component | Purpose | Notes |
|-----------|---------|-------|
| `SynthesisForm.jsx` | Handles dataset name, epsilon/delta, method selection, and file upload. | Relies on PapaParse for CSV validation; errors bubble up via local state. |
| `MetadataConfirmation.jsx` | Top-level container for the confirmation card, orchestrates per-column editors and validation. | Tracks validation issues in state; any red banner you see is triggered here. |
| `CategoricalEditor.jsx` | Lets users select/deselect categories, add custom values, and choose handling for excluded values. | Falls back to `value_counts`, `categories_preview`, etc., when the inferred list is empty, and blocks duplicate custom entries case-insensitively. |
| `NumericalEditor.jsx` | Edits min/max bounds, binning strategies, and DP budget sliders. | Non-numeric warnings show when the column couldn’t be parsed numerically. |
| `ResultsDisplay.jsx` | Presents the download button, preview table, and evaluation JSON block. | Shows inline warnings when evaluation fails and lets the user retry; download link is disabled while evaluation is in flight. |

## Component Tests

- `MetadataConfirmation.test.jsx` (Vitest) confirms that categorical lists render even when the backend only supplies `value_counts`, and it verifies the red alert banner blocks submission when categories are cleared or numeric bounds invert.
- `NumericalEditor.test.jsx` (Vitest) exercises the per-bound validation warnings (e.g., extremely large magnitudes and bad numeric input).
- Add further tests alongside their components in `frontend/src/components/`—Vitest picks up `*.test.jsx` files automatically.

## Manual Testing Checklist

- Upload a well-formed CSV and confirm that the metadata table is populated.
- Try an invalid CSV (e.g., mismatched columns) and ensure the form surfaces the error.
- On the confirmation page, clear all categories for a column and assert the validation banner appears; fix the error and verify submission succeeds.
- Enter an extremely large minimum/maximum (e.g., `1e12`) and verify the inline warning appears; try an invalid string to confirm it is rejected without updating the value.
- Attempt to add a custom category whose value already exists (e.g., `female` vs `Female`) and confirm the inline warning appears.
- Check number bounds (blank, non-numeric, max < min) all trigger errors.
- Confirm the results page renders both the preview and the evaluation JSON.

## Useful Commands

```bash
# Start the dev server
cd frontend
npm install
npm run dev -- --port 5174

# Lint
npm run lint

# Frontend end-to-end test (requires backend running)
E2E=1 pytest -q -k end_to_end
```

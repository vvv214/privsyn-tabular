import { useState } from 'react';

const STRATEGY_OPTIONS = [
  { value: 'uniform', label: 'Uniform bins (equal width)' },
  { value: 'exponential', label: 'Exponential bins (growing width)' },
  { value: 'dp_privtree', label: 'Differentially private (PrivTree)' },
];

const defaultBudgetFraction = 0.05;

const toFiniteNumber = (value) => {
  if (value === '' || value === null || value === undefined) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const NumericalEditor = ({ columnKey, domainDetails, onChange, showNonNumericWarning }) => {
  const bounds = domainDetails.bounds || {};
  const binning = domainDetails.binning || {};
  const [boundFeedback, setBoundFeedback] = useState({ min: null, max: null });
  const idBase = columnKey ? String(columnKey).replace(/[^a-zA-Z0-9_-]/g, '-').toLowerCase() : 'numeric';
  const minInputId = `${idBase}-min`;
  const maxInputId = `${idBase}-max`;
  const minTestId = `numeric-min-${idBase}`;
  const maxTestId = `numeric-max-${idBase}`;

  const updateBoundFeedback = (key, payload) => {
    setBoundFeedback((prev) => ({ ...prev, [key]: payload }));
  };

  const handleBoundsChange = (key, rawValue, isBadInput = false) => {
    if (isBadInput) {
      updateBoundFeedback(key, { level: 'error', text: 'Enter a finite number (e.g. 123 or 1.2e5).' });
      return;
    }
    if (rawValue === '' || rawValue === null || rawValue === undefined) {
      updateBoundFeedback(key, null);
      onChange({
        ...domainDetails,
        bounds: {
          ...bounds,
          [key]: '',
        },
      });
      return;
    }

    const trimmed = typeof rawValue === 'string' ? rawValue.trim() : rawValue;
    const numericValue = Number(trimmed);

    if (!Number.isFinite(numericValue)) {
      updateBoundFeedback(key, { level: 'error', text: 'Enter a finite number (e.g. 123 or 1.2e5).' });
      return;
    }

    if (Math.abs(numericValue) > 1e9) {
      updateBoundFeedback(key, { level: 'warning', text: 'Value is extremely large; double-check the expected units.' });
    } else {
      updateBoundFeedback(key, null);
    }

    onChange({
      ...domainDetails,
      bounds: {
        ...bounds,
        [key]: numericValue,
      },
    });
  };

  const handleBinningChange = (key, rawValue) => {
    let nextValue = rawValue;
    if (key !== 'method') {
      if (rawValue === '' || rawValue === null || rawValue === undefined) {
        nextValue = '';
      } else {
        const parsed = Number(rawValue);
        if (!Number.isFinite(parsed)) {
          return;
        }
        nextValue = parsed;
      }
    }
    onChange({
      ...domainDetails,
      binning: {
        ...binning,
        [key]: nextValue,
      },
    });
  };

  const handleBudgetChange = (event) => {
    const percent = toFiniteNumber(event.target.value) ?? 0;
    const fraction = Math.max(0, Math.min(100, percent)) / 100;
    handleBinningChange('dp_budget_fraction', fraction);
  };

  const budgetPercent = Math.round(100 * (binning.dp_budget_fraction ?? defaultBudgetFraction));

  let estimatedBinWidth = '';
  const binCount = toFiniteNumber(binning.bin_count);
  if (typeof bounds.min === 'number' && typeof bounds.max === 'number' && binCount && binCount > 0) {
    estimatedBinWidth = (bounds.max - bounds.min) / binCount;
  }

  return (
    <div className="mt-3">
      {showNonNumericWarning && (
        <div className="alert alert-warning" role="alert">
          Detected values for this column could not be parsed as numbers. Updating to numerical will coerce everything to the provided bounds.
        </div>
      )}
      <div className="row g-2">
        <div className="col-6">
          <label className="form-label" htmlFor={minInputId}>Minimum</label>
          <input
            type="number"
            step="any"
            className="form-control"
            value={bounds.min ?? ''}
            onChange={(e) => handleBoundsChange('min', e.target.value, e.target.validity?.badInput)}
            placeholder={domainDetails.numeric_summary?.min ?? ''}
            id={minInputId}
            data-testid={minTestId}
          />
          {boundFeedback.min && (
            <div className={`small mt-1 ${boundFeedback.min.level === 'error' ? 'text-danger' : 'text-warning'}`}>
              {boundFeedback.min.text}
            </div>
          )}
        </div>
        <div className="col-6">
          <label className="form-label" htmlFor={maxInputId}>Maximum</label>
          <input
            type="number"
            step="any"
            className="form-control"
            value={bounds.max ?? ''}
            onChange={(e) => handleBoundsChange('max', e.target.value, e.target.validity?.badInput)}
            placeholder={domainDetails.numeric_summary?.max ?? ''}
            id={maxInputId}
            data-testid={maxTestId}
          />
          {boundFeedback.max && (
            <div className={`small mt-1 ${boundFeedback.max.level === 'error' ? 'text-danger' : 'text-warning'}`}>
              {boundFeedback.max.text}
            </div>
          )}
        </div>
      </div>
      <div className="form-text">Values outside this range will be clipped before encoding.</div>

      <div className="row g-2 mt-3">
        <div className="col-6">
          <label className="form-label">Approximate bin width</label>
          <input
            type="number"
            step="any"
            className="form-control"
            value={binning.bin_width ?? ''}
            onChange={(e) => handleBinningChange('bin_width', e.target.value)}
            placeholder={estimatedBinWidth ? estimatedBinWidth.toFixed(4) : ''}
          />
        </div>
        <div className="col-6">
          <label className="form-label">Desired number of bins</label>
          <input
            type="number"
            className="form-control"
            value={binning.bin_count ?? ''}
            onChange={(e) => handleBinningChange('bin_count', e.target.value)}
            placeholder={domainDetails.size || 10}
          />
        </div>
      </div>
      <div className="form-text">We will reconcile bin width and count before discretization.</div>

      <div className="mt-3">
        <label className="form-label">Binning strategy</label>
        <select
          className="form-select"
          value={binning.method || 'uniform'}
          onChange={(e) => handleBinningChange('method', e.target.value)}
        >
          {STRATEGY_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </div>

      {binning.method === 'dp_privtree' && (
        <div className="mt-3">
          <label className="form-label">Privacy budget allocation ({budgetPercent}%)</label>
          <input
            type="range"
            className="form-range"
            min="1"
            max="25"
            value={Math.max(1, Math.min(25, budgetPercent))}
            onChange={handleBudgetChange}
          />
          <div className="form-text">Percentage of the global epsilon reserved for PrivTree inference.</div>
        </div>
      )}

      {binning.method === 'exponential' && (
        <div className="mt-3">
          <label className="form-label">Growth rate (optional)</label>
          <input
            type="number"
            step="any"
            className="form-control"
            value={binning.growth_rate ?? ''}
            onChange={(e) => handleBinningChange('growth_rate', e.target.value)}
            placeholder="e.g. 1.5"
          />
          <div className="form-text">Values larger than 1 lead to wider bins in the tail.</div>
        </div>
      )}
    </div>
  );
};

export default NumericalEditor;

const STRATEGY_OPTIONS = [
  { value: 'uniform', label: 'Uniform bins (equal width)' },
  { value: 'exponential', label: 'Exponential bins (growing width)' },
  { value: 'dp_privtree', label: 'Differentially private (PrivTree)' },
];

const defaultBudgetFraction = 0.05;

const parseNumber = (value, fallback = undefined) => {
  if (value === '' || value === null || value === undefined) return fallback;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? fallback : parsed;
};

const NumericalEditor = ({ domainDetails, onChange, showNonNumericWarning }) => {
  const bounds = domainDetails.bounds || {};
  const binning = domainDetails.binning || {};

  const handleBoundsChange = (key, rawValue) => {
    const nextValue = parseNumber(rawValue, rawValue === '' ? '' : 0);
    onChange({
      ...domainDetails,
      bounds: {
        ...bounds,
        [key]: nextValue,
      },
    });
  };

  const handleBinningChange = (key, rawValue) => {
    const nextValue = key === 'method' ? rawValue : parseNumber(rawValue, rawValue === '' ? '' : 0);
    onChange({
      ...domainDetails,
      binning: {
        ...binning,
        [key]: nextValue,
      },
    });
  };

  const handleBudgetChange = (event) => {
    const percent = parseNumber(event.target.value, 0);
    const fraction = Math.max(0, Math.min(100, percent)) / 100;
    handleBinningChange('dp_budget_fraction', fraction);
  };

  const budgetPercent = Math.round(100 * (binning.dp_budget_fraction ?? defaultBudgetFraction));

  let estimatedBinWidth = '';
  if (typeof bounds.min === 'number' && typeof bounds.max === 'number' && parseNumber(binning.bin_count) && parseNumber(binning.bin_count) > 0) {
    const count = parseNumber(binning.bin_count);
    if (count > 0) {
      estimatedBinWidth = (bounds.max - bounds.min) / count;
    }
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
          <label className="form-label">Minimum</label>
          <input
            type="number"
            step="any"
            className="form-control"
            value={bounds.min ?? ''}
            onChange={(e) => handleBoundsChange('min', e.target.value)}
            placeholder={domainDetails.numeric_summary?.min ?? ''}
          />
        </div>
        <div className="col-6">
          <label className="form-label">Maximum</label>
          <input
            type="number"
            step="any"
            className="form-control"
            value={bounds.max ?? ''}
            onChange={(e) => handleBoundsChange('max', e.target.value)}
            placeholder={domainDetails.numeric_summary?.max ?? ''}
          />
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

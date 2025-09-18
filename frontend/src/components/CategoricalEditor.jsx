import { useMemo, useState } from 'react';

const SPECIAL_TOKEN_DEFAULT = '__OTHER__';

const CategoricalEditor = ({
  columnKey,
  domainDetails,
  onChange,
}) => {
  const {
    categories = [],
    selected_categories: selected = [],
    custom_categories: custom = [],
    value_counts: counts = {},
    category_null_token: nullToken,
    excluded_strategy: strategy = 'map_to_special',
    special_token: specialToken = SPECIAL_TOKEN_DEFAULT,
  } = domainDetails;

  const [newCategory, setNewCategory] = useState('');
  const [helperMessage, setHelperMessage] = useState(null);
  const idBase = columnKey ? String(columnKey).replace(/[^a-zA-Z0-9_-]/g, '-').toLowerCase() : 'categorical';
  const selectTestId = `categorical-select-${idBase}`;
  const clearTestId = `categorical-clear-${idBase}`;
  const selectAllTestId = `categorical-selectall-${idBase}`;

  const calculateSize = (selectedValues, customValues) => {
    const uniq = new Set([...(selectedValues || []), ...(customValues || [])]);
    return uniq.size;
  };

  const availableCategories = useMemo(() => {
    const uniq = new Set();

    const addValues = (vals) => {
      if (!Array.isArray(vals)) return;
      vals.forEach((val) => {
        if (val === undefined || val === null) return;
        const strVal = String(val);
        if (strVal.length === 0) return;
        uniq.add(strVal);
      });
    };

    addValues(categories);
    addValues(domainDetails?.categories_preview);
    addValues(domainDetails?.categories_from_data);
    addValues(custom);
    Object.keys(counts || {}).forEach((key) => {
      if (key !== undefined && key !== null && String(key).length > 0) {
        uniq.add(String(key));
      }
    });

    if (uniq.size === 0) {
      addValues(selected);
    }

    return Array.from(uniq);
  }, [categories, custom, counts, domainDetails, selected]);

  const appliedSelected = useMemo(() => {
    const selectedSet = new Set(selected);
    custom.forEach((val) => {
      if (!selectedSet.has(val)) {
        selectedSet.add(val);
      }
    });
    return Array.from(selectedSet).filter((val) => availableCategories.includes(val));
  }, [custom, selected, availableCategories]);

  const excludedValues = useMemo(() => {
    const selectedSet = new Set(appliedSelected);
    return availableCategories.filter((val) => !selectedSet.has(val));
  }, [availableCategories, appliedSelected]);

  const clearHelper = () => setHelperMessage(null);

  const handleSelectionChange = (event) => {
    const options = Array.from(event.target.selectedOptions || []);
    const updatedSelected = options.map((opt) => opt.value);
    clearHelper();
    onChange({
      ...domainDetails,
      selected_categories: updatedSelected,
      size: calculateSize(updatedSelected, custom),
    });
  };

  const handleSelectAll = () => {
    clearHelper();
    onChange({
      ...domainDetails,
      selected_categories: availableCategories,
      size: calculateSize(availableCategories, custom),
    });
  };

  const handleClearAll = () => {
    clearHelper();
    onChange({
      ...domainDetails,
      selected_categories: [],
      size: calculateSize([], custom),
    });
  };

  const handleAddCategory = () => {
    const trimmed = newCategory.trim();
    if (!trimmed) {
      setHelperMessage('Enter a non-empty value before adding.');
      return;
    }

    const normalizedValue = trimmed.toLowerCase();
    const normalizedExisting = new Set(availableCategories.map((val) => String(val).toLowerCase()));

    if (normalizedExisting.has(normalizedValue)) {
      setHelperMessage(`"${trimmed}" already exists in the category list.`);
      setNewCategory('');
      return;
    }

    if (availableCategories.includes(trimmed)) {
      if (!appliedSelected.includes(trimmed)) {
        onChange({
          ...domainDetails,
          selected_categories: [...appliedSelected, trimmed],
        });
      }
      setNewCategory('');
      return;
    }
    const updatedCustom = [...custom, trimmed];
    onChange({
      ...domainDetails,
      custom_categories: updatedCustom,
      selected_categories: [...appliedSelected, trimmed],
      size: calculateSize([...appliedSelected, trimmed], updatedCustom),
    });
    setHelperMessage(null);
    setNewCategory('');
  };

  const handleStrategyChange = (event) => {
    clearHelper();
    onChange({
      ...domainDetails,
      excluded_strategy: event.target.value,
    });
  };

  const handleSpecialTokenChange = (event) => {
    const updatedToken = event.target.value;
    clearHelper();
    onChange({
      ...domainDetails,
      special_token: updatedToken || SPECIAL_TOKEN_DEFAULT,
    });
  };

  return (
    <div className="mt-3">
      <label className="form-label">Values detected ({availableCategories.length})</label>
      <div className="mb-2 d-flex gap-2 flex-wrap">
        <button
          type="button"
          className="btn btn-sm btn-outline-primary"
          onClick={handleSelectAll}
          data-testid={selectAllTestId}
        >
          Select all
        </button>
        <button
          type="button"
          className="btn btn-sm btn-outline-secondary"
          onClick={handleClearAll}
          data-testid={clearTestId}
        >
          Clear all
        </button>
        {nullToken && appliedSelected.length !== availableCategories.length && (
          <span className="badge bg-light text-dark">Missing token will map to {nullToken}</span>
        )}
      </div>
      <select
        multiple
        size={Math.min(8, Math.max(availableCategories.length, 4))}
        className="form-select"
        value={appliedSelected}
        onChange={handleSelectionChange}
        data-testid={selectTestId}
      >
        {availableCategories.map((val) => (
          <option
            key={val}
            value={val}
            style={appliedSelected.includes(val) ? { backgroundColor: '#0d6efd', color: '#fff' } : undefined}
          >
            {val} (count: {counts?.[val] ?? counts?.[String(val)] ?? 0})
          </option>
        ))}
      </select>
      <div className="form-text">Unselect values to mark them as unexpected in the uploaded data.</div>

      <div className="mt-3">
        <p className="mb-1 fw-semibold">Selected values ({appliedSelected.length})</p>
        {appliedSelected.length ? (
          <div className="d-flex flex-wrap gap-2">
            {appliedSelected.map((val) => (
              <span key={`sel-${val}`} className="badge bg-primary-subtle text-dark border border-primary-subtle">
                {val}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-muted">No values selected. Use the list above to include categories.</p>
        )}
      </div>

      <div className="mt-3">
        <p className="mb-1 fw-semibold">Available to exclude ({excludedValues.length})</p>
        {excludedValues.length ? (
          <div className="d-flex flex-wrap gap-2">
            {excludedValues.map((val) => (
              <span key={`ex-${val}`} className="badge bg-light text-dark border">
                {val}
              </span>
            ))}
          </div>
        ) : (
          <p className="text-muted">All detected values are currently selected.</p>
        )}
      </div>

      <div className="input-group input-group-sm mt-3">
        <input
          type="text"
          className="form-control"
          placeholder="Add new category"
          value={newCategory}
          onChange={(e) => {
            setNewCategory(e.target.value);
            setHelperMessage(null);
          }}
        />
        <button type="button" className="btn btn-outline-success" onClick={handleAddCategory}>Add</button>
      </div>
      <div className="form-text">New values will be included in the domain even if they do not exist in the original data.</div>

      {helperMessage && (
        <div className="text-danger small mt-2" data-testid="category-helper-message">
          {helperMessage}
        </div>
      )}

      {excludedValues.length > 0 && (
        <div className="alert alert-warning mt-3" role="alert">
          {excludedValues.length} value(s) appear in the data but are currently excluded: {excludedValues.slice(0, 5).join(', ')}{excludedValues.length > 5 ? ' …' : ''}
        </div>
      )}

      <div className="mt-3">
        <label className="form-label">Unexpected value handling</label>
        <select className="form-select" value={strategy} onChange={handleStrategyChange}>
          <option value="map_to_special">Map excluded values to a special token</option>
          <option value="resample">Change to another in-domain value</option>
        </select>
       {strategy === 'map_to_special' && (
         <div className="mt-2">
           <label className="form-label">Special token</label>
           <input
             type="text"
             className="form-control"
             value={specialToken}
             onChange={handleSpecialTokenChange}
             placeholder={SPECIAL_TOKEN_DEFAULT}
           />
           <div className="form-text">All excluded values will be replaced with this token before encoding.</div>
         </div>
        )}
        {strategy === 'resample' && (
          <div className="mt-2 text-muted small">
            Excluded values will be reassigned randomly to one of the selected categories before encoding.
          </div>
        )}
      </div>
    </div>
  );
};

export default CategoricalEditor;

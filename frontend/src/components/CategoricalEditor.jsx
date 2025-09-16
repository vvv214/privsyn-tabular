import { useMemo, useState } from 'react';

const SPECIAL_TOKEN_DEFAULT = '__OTHER__';

const CategoricalEditor = ({
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

  const calculateSize = (selectedValues, customValues) => {
    const uniq = new Set([...(selectedValues || []), ...(customValues || [])]);
    return uniq.size;
  };

  const availableCategories = useMemo(() => {
    const base = Array.isArray(categories) && categories.length > 0
      ? categories
      : Object.keys(counts || {});
    const uniq = new Set(base);
    custom.forEach((val) => {
      if (val && !uniq.has(val)) {
        uniq.add(val);
      }
    });
    return Array.from(uniq);
  }, [categories, custom]);

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

  const handleSelectionChange = (event) => {
    const options = Array.from(event.target.selectedOptions || []);
    const updatedSelected = options.map((opt) => opt.value);
    onChange({
      ...domainDetails,
      selected_categories: updatedSelected,
      size: calculateSize(updatedSelected, custom),
    });
  };

  const handleSelectAll = () => {
    onChange({
      ...domainDetails,
      selected_categories: availableCategories,
      size: calculateSize(availableCategories, custom),
    });
  };

  const handleClearAll = () => {
    onChange({
      ...domainDetails,
      selected_categories: [],
      size: calculateSize([], custom),
    });
  };

  const handleAddCategory = () => {
    const trimmed = newCategory.trim();
    if (!trimmed) return;
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
    setNewCategory('');
  };

  const handleStrategyChange = (event) => {
    onChange({
      ...domainDetails,
      excluded_strategy: event.target.value,
    });
  };

  const handleSpecialTokenChange = (event) => {
    const updatedToken = event.target.value;
    onChange({
      ...domainDetails,
      special_token: updatedToken || SPECIAL_TOKEN_DEFAULT,
    });
  };

  return (
    <div className="mt-3">
      <label className="form-label">Values detected ({availableCategories.length})</label>
      <div className="mb-2 d-flex gap-2 flex-wrap">
        <button type="button" className="btn btn-sm btn-outline-primary" onClick={handleSelectAll}>Select all</button>
        <button type="button" className="btn btn-sm btn-outline-secondary" onClick={handleClearAll}>Clear all</button>
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
      >
        {availableCategories.map((val) => (
          <option
            key={val}
            value={val}
            style={appliedSelected.includes(val) ? { backgroundColor: '#0d6efd', color: '#fff' } : undefined}
          >
            {val} (count: {counts[val] ?? 0})
          </option>
        ))}
      </select>
      <div className="form-text">Unselect values to mark them as unexpected in the uploaded data.</div>

      <div className="input-group input-group-sm mt-3">
        <input
          type="text"
          className="form-control"
          placeholder="Add new category"
          value={newCategory}
          onChange={(e) => setNewCategory(e.target.value)}
        />
        <button type="button" className="btn btn-outline-success" onClick={handleAddCategory}>Add</button>
      </div>
      <div className="form-text">New values will be included in the domain even if they do not exist in the original data.</div>

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

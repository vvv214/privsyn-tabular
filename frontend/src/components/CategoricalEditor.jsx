import { useMemo, useState } from 'react';

const CategoricalEditor = ({
  columnKey,
  domainDetails,
  onChange,
}) => {
  const {
    categories = [],
    selected_categories: selected = [],
    custom_categories: custom = [],
    category_null_token: nullToken,
    categories_preview: categoriesPreview = [],
    categories_from_data: categoriesFromData = [],
  } = domainDetails;
  const [newCategory, setNewCategory] = useState('');
  const [helperMessage, setHelperMessage] = useState(null);
  const idBase = columnKey ? String(columnKey).replace(/[^a-zA-Z0-9_-]/g, '-').toLowerCase() : 'categorical';
  const selectTestId = `categorical-select-${idBase}`;

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
    addValues(categoriesPreview);
    addValues(categoriesFromData);
    addValues(custom);

    if (uniq.size === 0) {
      addValues(selected);
    }

    return Array.from(uniq);
  }, [categories, categoriesPreview, categoriesFromData, custom, selected]);

  const appliedSelected = useMemo(() => {
    const selectedSet = new Set(selected);
    custom.forEach((val) => {
      if (!selectedSet.has(val)) {
        selectedSet.add(val);
      }
    });
    return Array.from(selectedSet).filter((val) => availableCategories.includes(val));
  }, [custom, selected, availableCategories]);

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

  const handleRemoveCategory = (value) => {
    const normalizedValue = String(value);
    if (!availableCategories.includes(normalizedValue)) {
      return;
    }

    // Removing detected categories is discouraged; reminder already exists in surrounding text.
    // We allow removal only for values that were added manually.
    if (!custom.includes(normalizedValue)) {
      setHelperMessage(`"${normalizedValue}" is part of the detected domain. Adjust upstream data if you need to exclude it.`);
      return;
    }

    const updatedCustom = custom.filter((val) => val !== normalizedValue);
    const updatedSelected = appliedSelected.filter((val) => val !== normalizedValue);

    onChange({
      ...domainDetails,
      custom_categories: updatedCustom,
      selected_categories: updatedSelected,
      size: calculateSize(updatedSelected, updatedCustom),
    });
  };

  return (
    <div className="mt-3">
      <div className="form-text mb-2">
        Define the domain based on your knowledge of the data. Even if certain categories are absent in this sample,
        include the full set you expect in production to keep the overall process DP.
      </div>
      <label className="form-label">Values detected ({availableCategories.length})</label>
      <div className="mb-2 d-flex gap-2 flex-wrap">
        {nullToken && (
          <span className="badge bg-light text-dark">Missing token will map to {nullToken}</span>
        )}
      </div>
      <div className="d-flex flex-wrap gap-2" data-testid={selectTestId}>
        {availableCategories.length > 0 ? (
          availableCategories.map((val) => {
            const normalizedValue = String(val);
            const isCustom = custom.includes(normalizedValue);
            return (
              <span
                key={normalizedValue}
                className="badge bg-primary-subtle text-dark border border-primary-subtle d-inline-flex align-items-center gap-2"
                style={{ padding: '0.45rem 0.75rem' }}
              >
                {normalizedValue}
                {isCustom && (
                  <button
                    type="button"
                    className="btn btn-sm btn-link p-0"
                    aria-label={`Remove ${normalizedValue}`}
                    onClick={() => handleRemoveCategory(normalizedValue)}
                    style={{ color: 'rgba(55, 48, 163, 0.85)', textDecoration: 'none' }}
                  >
                    ×
                  </button>
                )}
              </span>
            );
          })
        ) : (
          <span className="text-muted">No detected values.</span>
        )}
      </div>
      <div className="form-text">All detected values remain in the domain. Adjust the dataset upstream if any should be excluded.</div>

      <div className="input-group input-group-sm mt-3">
        <input
          type="text"
          className="form-control"
          placeholder="Add new category"
          data-testid={`category-input-${idBase}`}
          value={newCategory}
          onChange={(e) => {
            setNewCategory(e.target.value);
            setHelperMessage(null);
          }}
        />
      <button
        type="button"
        className="btn btn-outline-success"
        data-testid={`category-add-${idBase}`}
        onClick={handleAddCategory}
      >
        Add
      </button>
      </div>

      {helperMessage && (
        <div className="text-danger small mt-2" data-testid="category-helper-message">
          {helperMessage}
        </div>
      )}
    </div>
  );
};

export default CategoricalEditor;

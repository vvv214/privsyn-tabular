import React, { useState, useEffect } from 'react';
import CategoricalEditor from './CategoricalEditor';
import NumericalEditor from './NumericalEditor';

const SPECIAL_TOKEN_DEFAULT = '__OTHER__';

const MetadataConfirmation = ({ uniqueId, inferredDomainData, inferredInfoData, onConfirm, onCancel, isSubmitting = false }) => {
    const [domainData, setDomainData] = useState({});
    const [infoData, setInfoData] = useState(inferredInfoData);
    const [validationMessages, setValidationMessages] = useState([]);

    const inferenceSettings = infoData?.inference_settings || {};
    const inferenceReport = infoData?.inference_report || {};

    const normalizeDomainData = (rawDomain) => {
        const normalized = {};
        Object.entries(rawDomain || {}).forEach(([column, details]) => {
            if (details.type === 'categorical') {
                const fallbackCategories = Array.isArray(details.categories_preview)
                    ? [...details.categories_preview]
                    : Object.keys(details.value_counts || {});
                const baseCategories = Array.isArray(details.categories) && details.categories.length > 0
                    ? [...details.categories]
                    : fallbackCategories;
                const categories = baseCategories;

                const selected = [...categories];
                const custom = [];
                const normalizedStrategy = details.excluded_strategy === 'keep_in_domain'
                    ? 'resample'
                    : details.excluded_strategy || 'map_to_special';
                const normalizedEntry = {
                    ...details,
                    categories,
                    selected_categories: selected,
                    custom_categories: custom,
                    excluded_strategy: normalizedStrategy,
                    special_token: details.special_token || SPECIAL_TOKEN_DEFAULT,
                    numeric_candidate_summary: details.numeric_candidate_summary || null,
                    nonNumericWarning: false,
                };
                normalizedEntry.size = new Set([...(normalizedEntry.selected_categories || []), ...(normalizedEntry.custom_categories || [])]).size || details.size || categories.length;
                normalized[column] = normalizedEntry;
            } else {
                const summary = details.numeric_summary || {};
                const bounds = {
                    min: details.bounds?.min ?? summary.min ?? '',
                    max: details.bounds?.max ?? summary.max ?? '',
                };
                const defaultBinCount = details.binning?.bin_count ?? Math.min(20, Math.max(5, details.size || 10));
                const normalizedEntry = {
                    ...details,
                    bounds,
                    binning: {
                        method: details.binning?.method || 'uniform',
                        bin_count: defaultBinCount,
                        bin_width: details.binning?.bin_width ?? '',
                        growth_rate: details.binning?.growth_rate ?? '',
                        dp_budget_fraction: details.binning?.dp_budget_fraction ?? 0.05,
                    },
                    numeric_candidate_summary: details.numeric_candidate_summary || details.numeric_summary || null,
                    nonNumericWarning: false,
                };
                normalized[column] = normalizedEntry;
            }
        });
        return normalized;
    };

    useEffect(() => {
        const normalized = normalizeDomainData(inferredDomainData);
        setDomainData(normalized);

        const numColumns = Object.entries(normalized).filter(([, entry]) => entry.type === 'numerical').map(([columnName]) => columnName);
        const catColumns = Object.entries(normalized).filter(([, entry]) => entry.type === 'categorical').map(([columnName]) => columnName);

        setInfoData(prevInfoData => ({
            ...prevInfoData,
            ...inferredInfoData, // Apply incoming info data first
            num_columns: numColumns,
            cat_columns: catColumns,
            n_num_features: numColumns.length,
            n_cat_features: catColumns.length,
        }));
    }, [inferredDomainData, inferredInfoData]);

    

    const handleDomainChange = (key, subKey, value) => {
        setValidationMessages([]);
        setDomainData(prev => {
            const previousEntry = prev[key] || {};
            let updatedEntry = { ...previousEntry };

            if (subKey === 'type') {
                const snapshot = previousEntry._preserved_categorical || {
                    categories: Array.isArray(previousEntry.categories) ? [...previousEntry.categories] : [],
                    selected_categories: Array.isArray(previousEntry.selected_categories) ? [...previousEntry.selected_categories] : [],
                    custom_categories: Array.isArray(previousEntry.custom_categories) ? [...previousEntry.custom_categories] : [],
                    excluded_strategy: previousEntry.excluded_strategy,
                    special_token: previousEntry.special_token,
                    value_counts: previousEntry.value_counts,
                    categories_preview: previousEntry.categories_preview,
                    category_null_token: previousEntry.category_null_token,
                    size: previousEntry.size,
                };
                updatedEntry.type = value;
                if (value === 'categorical') {
                    const fallbackCategories = Array.isArray(previousEntry.categories_preview)
                        ? previousEntry.categories_preview
                        : Object.keys(previousEntry.value_counts || {});
                    const baseCategories = snapshot.categories && snapshot.categories.length > 0
                        ? snapshot.categories
                        : fallbackCategories;
                    const categories = baseCategories;
                    const selectedSnapshot = snapshot.selected_categories && snapshot.selected_categories.length > 0
                        ? snapshot.selected_categories
                        : categories;
                    const customSnapshot = Array.isArray(snapshot.custom_categories) ? snapshot.custom_categories : [];
                    updatedEntry = {
                        ...updatedEntry,
                        categories,
                        selected_categories: [...selectedSnapshot],
                        custom_categories: [...customSnapshot],
                        excluded_strategy: snapshot.excluded_strategy || 'map_to_special',
                        special_token: snapshot.special_token || SPECIAL_TOKEN_DEFAULT,
                        value_counts: snapshot.value_counts || updatedEntry.value_counts,
                        categories_preview: snapshot.categories_preview || updatedEntry.categories_preview,
                        category_null_token: snapshot.category_null_token || updatedEntry.category_null_token,
                        size: new Set([...selectedSnapshot, ...customSnapshot]).size || (snapshot.size ?? categories.length),
                        nonNumericWarning: false,
                        _preserved_categorical: undefined,
                    };
                } else {
                    const candidateSummary = previousEntry.numeric_candidate_summary || previousEntry.numeric_summary;
                    const hasCandidate = candidateSummary && candidateSummary.min !== null && candidateSummary.max !== null;
                    updatedEntry = {
                        ...updatedEntry,
                        custom_categories: [],
                        selected_categories: [],
                        bounds: {
                            min: previousEntry.bounds?.min ?? (hasCandidate ? candidateSummary.min : ''),
                            max: previousEntry.bounds?.max ?? (hasCandidate ? candidateSummary.max : ''),
                        },
                        binning: {
                            method: previousEntry.binning?.method || 'uniform',
                            bin_count: previousEntry.binning?.bin_count ?? Math.min(20, Math.max(5, previousEntry.size || 10)),
                            bin_width: previousEntry.binning?.bin_width ?? '',
                            growth_rate: previousEntry.binning?.growth_rate ?? '',
                            dp_budget_fraction: previousEntry.binning?.dp_budget_fraction ?? 0.05,
                        },
                        nonNumericWarning: !hasCandidate,
                        _preserved_categorical: snapshot,
                    };
                }
            } else {
                updatedEntry = {
                    ...updatedEntry,
                    [subKey]: parseFloat(value) || 0
                };
            }

            const updatedDomain = {
                ...prev,
                [key]: updatedEntry
            };

            // After changing the type, we need to recalculate the counts for n_num_features and n_cat_features
            if (subKey === 'type') {
                let num_count = 0;
                let cat_count = 0;
                const num_columns = [];
                const cat_columns = [];
                Object.entries(updatedDomain).forEach(([columnName, entry]) => {
                    if (entry.type === 'numerical') {
                        num_count++;
                        num_columns.push(columnName);
                    } else if (entry.type === 'categorical') {
                        cat_count++;
                        cat_columns.push(columnName);
                    }
                });
                setInfoData(prevInfo => ({
                    ...prevInfo,
                    n_num_features: num_count,
                    n_cat_features: cat_count,
                    num_columns,
                    cat_columns
                }));
            }
            return updatedDomain;
        });
    };

    const sanitizeNumber = (value) => {
        if (value === '' || value === null || value === undefined) {
            return null;
        }
        const asNumber = Number(value);
        if (Number.isNaN(asNumber) || !Number.isFinite(asNumber)) {
            return null;
        }
        return asNumber;
    };

    const validateDomainData = (domain) => {
        const messages = [];
        Object.entries(domain).forEach(([column, details]) => {
            if (details.type === 'categorical') {
                const selectedCount = (details.selected_categories?.length || 0) + (details.custom_categories?.length || 0);
                if (selectedCount === 0) {
                    messages.push(`Select at least one category for "${column}".`);
                }
            } else if (details.type === 'numerical') {
                const min = sanitizeNumber(details.bounds?.min);
                const max = sanitizeNumber(details.bounds?.max);
                if (min !== null && max !== null && max < min) {
                    messages.push(`Maximum must be greater than or equal to minimum for "${column}".`);
                }
            }
        });
        return messages;
    };

    const buildConfirmedDomainData = () => {
        const serialized = {};
        Object.entries(domainData).forEach(([column, details]) => {
            if (details.type === 'categorical') {
                const categoriesFromData = Array.isArray(details.categories) ? details.categories : [];
                const selected = Array.isArray(details.selected_categories) && details.selected_categories.length > 0
                    ? details.selected_categories
                    : categoriesFromData;
                const custom = Array.isArray(details.custom_categories) ? details.custom_categories : [];
                const excluded = categoriesFromData.filter((cat) => !selected.includes(cat));
                const specialToken = details.special_token || SPECIAL_TOKEN_DEFAULT;
                let finalCategories;
                if (details.excluded_strategy === 'resample') {
                    finalCategories = Array.from(new Set([...selected, ...custom]));
                } else {
                    finalCategories = Array.from(new Set([...selected, ...custom, specialToken]));
                }
                serialized[column] = {
                    type: 'categorical',
                    categories_from_data: categoriesFromData,
                    selected_categories: selected,
                    custom_categories: custom,
                    excluded_categories: excluded,
                    excluded_strategy: details.excluded_strategy || 'map_to_special',
                    special_token: specialToken,
                    category_null_token: details.category_null_token,
                    value_counts: details.value_counts || {},
                    categories: finalCategories,
                    size: finalCategories.length,
                    numeric_candidate_summary: details.numeric_candidate_summary,
                };
            } else {
                const bounds = {
                    min: sanitizeNumber(details.bounds?.min),
                    max: sanitizeNumber(details.bounds?.max),
                };
                const binning = {
                    method: details.binning?.method || 'uniform',
                    bin_count: sanitizeNumber(details.binning?.bin_count),
                    bin_width: sanitizeNumber(details.binning?.bin_width),
                    growth_rate: sanitizeNumber(details.binning?.growth_rate),
                    dp_budget_fraction: details.binning?.dp_budget_fraction ?? 0.05,
                };
                serialized[column] = {
                    type: 'numerical',
                    bounds,
                    binning,
                    numeric_summary: details.numeric_summary,
                    numeric_candidate_summary: details.numeric_candidate_summary,
                    size: details.size,
                };
            }
        });
        return serialized;
    };

    const handleInfoChange = (key, value) => {
        setInfoData(prev => ({ ...prev, [key]: value }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        const messages = validateDomainData(domainData);
        if (messages.length > 0) {
            setValidationMessages(messages);
            setTimeout(() => {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }, 0);
            return;
        }

        setValidationMessages([]);
        const serializedDomain = buildConfirmedDomainData();
        const nextInfo = {
            ...infoData,
            num_columns: Object.entries(serializedDomain).filter(([, details]) => details.type === 'numerical').map(([col]) => col),
            cat_columns: Object.entries(serializedDomain).filter(([, details]) => details.type === 'categorical').map(([col]) => col),
        };
        onConfirm(uniqueId, serializedDomain, nextInfo);
    };

    return (
        <div className="card shadow-sm">
            <div className="card-header bg-info text-white">
                <h3 className="mb-0">Confirm Inferred Metadata</h3>
            </div>
            <div className="card-body">
                <form onSubmit={handleSubmit}>
                    {validationMessages.length > 0 && (
                        <div className="alert alert-danger" role="alert" data-testid="metadata-validation-alert">
                            <p className="mb-2 fw-bold">Please address the following issues:</p>
                            <ul className="mb-0">
                                {validationMessages.map((message, index) => (
                                    <li key={`${message}-${index}`}>{message}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                    <div className="card mb-4">
                        <div className="card-header">
                            <h4 className="mb-0">Dataset Information</h4>
                        </div>
                        <div className="card-body">
                            <div className="alert alert-info" role="alert">
                                <strong>How did we infer types?</strong> Numerical integer columns with fewer than {inferenceSettings.integer_unique_threshold ?? '...'} unique values, or sparse integers with fewer than {Math.round((inferenceSettings.integer_unique_ratio_threshold ?? 0) * 100)}% uniques (max {inferenceSettings.integer_unique_max ?? '...'}) are treated as categorical—for example, <code>age</code> may become categorical when only a handful of ages appear in the uploaded sample. Float columns need at least {inferenceSettings.float_unique_threshold ?? '...'} unique values to remain numerical.
                            </div>
                            <div className="row">
                                <div className="col-md-6 mb-3 d-flex justify-content-center align-items-center flex-column">
                                    <label htmlFor="info_name" className="form-label">Name</label>
                                    <input type="text" id="info_name" className="form-control" value={infoData.name || ''} onChange={(e) => handleInfoChange('name', e.target.value)} />
                                </div>
                                <div className="col-md-6 mb-3 d-flex justify-content-center align-items-center flex-column">
                                    <label htmlFor="info_n_num_features" className="form-label">Numerical Features</label>
                                    <input type="number" id="info_n_num_features" className="form-control" value={infoData.n_num_features || 0} readOnly />
                                </div>
                                <div className="col-md-6 mb-3 d-flex justify-content-center align-items-center flex-column">
                                    <label htmlFor="info_n_cat_features" className="form-label">Categorical Features</label>
                                    <input type="number" id="info_n_cat_features" className="form-control" value={infoData.n_cat_features || 0} readOnly />
                                </div>
                                <div className="col-md-6 mb-3 d-flex justify-content-center align-items-center flex-column">
                                    <label htmlFor="info_train_size" className="form-label">Dataset Size</label>
                                    <input type="number" id="info_train_size" className="form-control" value={infoData.train_size || 0} onChange={(e) => handleInfoChange('train_size', parseInt(e.target.value) || 0)} />
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-header">
                            <h4 className="mb-0">Domain Information</h4>
                        </div>
                        <div className="card-body">
                            <div className="alert alert-warning" role="alert">
                                <strong>Note:</strong> The domain info below is inferred by looking at the data to improve usability. Strictly speaking, this is not differentially private. To satisfy DP, these domain settings should be specified data-agnostically using domain knowledge.
                            </div>
                            <div className="row">
                                {Object.entries(domainData).map(([key, value]) => (
                                    <div className="col-md-6 mb-4" key={key}>
                                        <div className="card h-100">
                                            <div className="card-body">
                                                <h5 className="card-title text-center mb-3">{key}</h5>
                                                <p className="text-muted small">
                                                    {inferenceReport[key]?.reasons?.map((reason, idx) => (
                                                        <span key={reason.code || idx} className="d-block">- {reason.message}</span>
                                                    )) || 'Reasoning unavailable.'}
                                                </p>
                                                <div className="mb-2">
                                                    <label htmlFor={`domain_${key}_type`} className="form-label">Type</label>
                                                    {(() => {
                                                        const reasons = inferenceReport[key]?.reasons || [];
                                                        const reasonCodes = new Set(reasons.map((reason) => reason.code));
                                                        const ambiguousCodes = new Set([
                                                            'integer_unique_below_threshold',
                                                            'integer_sparse_unique',
                                                            'float_unique_low',
                                                        ]);
                                                        const allowNumericOption = Array.from(ambiguousCodes).some((code) => reasonCodes.has(code));
                                                        const typeOptions = value.type === 'categorical'
                                                            ? ['categorical', ...(allowNumericOption ? ['numerical'] : [])]
                                                            : ['numerical', ...(allowNumericOption ? ['categorical'] : [])];

                                                        return (
                                                            <select
                                                                id={`domain_${key}_type`}
                                                                className="form-select"
                                                                value={value.type}
                                                                onChange={(e) => handleDomainChange(key, 'type', e.target.value)}
                                                            >
                                                                {typeOptions.map((typeOption) => (
                                                                    <option key={typeOption} value={typeOption}>
                                                                        {typeOption === 'categorical' ? 'Categorical' : 'Numerical'}
                                                                    </option>
                                                                ))}
                                                            </select>
                                                        );
                                                    })()}
                                                </div>
                                                {value.type === 'categorical' ? (
                                                    <CategoricalEditor
                                                        columnKey={key}
                                                        domainDetails={value}
                                                        onChange={(updatedDetails) => {
                                                            setDomainData(prevDomain => ({
                                                                ...prevDomain,
                                                                [key]: {
                                                                    ...prevDomain[key],
                                                                    ...updatedDetails,
                                                                }
                                                            }));
                                                        }}
                                                    />
                                                ) : (
                                                    <NumericalEditor
                                                        columnKey={key}
                                                        domainDetails={value}
                                                        showNonNumericWarning={value.nonNumericWarning}
                                                        onChange={(updatedDetails) => {
                                                            setDomainData(prevDomain => ({
                                                                ...prevDomain,
                                                                [key]: {
                                                                    ...prevDomain[key],
                                                                    ...updatedDetails,
                                                                }
                                                            }));
                                                        }}
                                                    />
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    <div className="d-flex justify-content-end mt-4">
                        <button type="button" className="btn btn-secondary me-2" onClick={onCancel}>Cancel</button>
                        <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
                            {isSubmitting ? 'Submitting…' : 'Confirm & Synthesize'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default MetadataConfirmation;

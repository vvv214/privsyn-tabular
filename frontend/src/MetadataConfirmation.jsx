import React, { useState, useEffect } from 'react';

const MetadataConfirmation = ({ uniqueId, inferredDomainData, inferredInfoData, synthesisParams, onConfirm, onCancel }) => {
    const [domainData, setDomainData] = useState(inferredDomainData);
    const [infoData, setInfoData] = useState(inferredInfoData);

    useEffect(() => {
        setDomainData(inferredDomainData);
        setInfoData(inferredInfoData);
    }, [inferredDomainData, inferredInfoData]);

    const [columnNameMap, setColumnNameMap] = useState({});

    useEffect(() => {
        const map = {};
        if (inferredInfoData) {
            inferredInfoData.num_columns.forEach((colName, index) => {
                map[`num_attr_${index + 1}`] = colName;
            });
            inferredInfoData.cat_columns.forEach((colName, index) => {
                map[`cat_attr_${index + 1}`] = colName;
            });
        }
        setColumnNameMap(map);
    }, [inferredInfoData]);

    const handleDomainChange = (key, value, subKey = null) => {
        setDomainData(prev => {
            if (subKey) { // For numerical min/max
                return {
                    ...prev,
                    [key]: {
                        ...prev[key],
                        [subKey]: parseFloat(value) || 0
                    }
                };
            } else { // For categorical
                return { ...prev, [key]: parseInt(value) || 0 };
            }
        });
    };

    const handleInfoChange = (key, value) => {
        setInfoData(prev => ({ ...prev, [key]: value }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onConfirm(uniqueId, domainData, infoData);
    };

    return (
        <div className="card shadow-sm">
            <div className="card-header bg-info text-white">
                <h3 className="mb-0">Confirm Inferred Metadata</h3>
            </div>
            <div className="card-body">
                <form onSubmit={handleSubmit}>
                    <div className="card mb-4">
                        <div className="card-header">
                            <h4 className="mb-0">Dataset Information</h4>
                        </div>
                        <div className="card-body">
                            <div className="row">
                                <div className="col-md-6 mb-3 d-flex justify-content-center align-items-center flex-column">
                                    <label htmlFor="info_name" className="form-label">Name</label>
                                    <input type="text" id="info_name" className="form-control" value={infoData.name || ''} onChange={(e) => handleInfoChange('name', e.target.value)} />
                                </div>
                                <div className="col-md-6 mb-3 d-flex justify-content-center align-items-center flex-column">
                                    <label htmlFor="info_n_num_features" className="form-label">Numerical Features</label>
                                    <input type="number" id="info_n_num_features" className="form-control" value={infoData.n_num_features || 0} onChange={(e) => handleInfoChange('n_num_features', parseInt(e.target.value) || 0)} />
                                </div>
                                <div className="col-md-6 mb-3 d-flex justify-content-center align-items-center flex-column">
                                    <label htmlFor="info_n_cat_features" className="form-label">Categorical Features</label>
                                    <input type="number" id="info_n_cat_features" className="form-control" value={infoData.n_cat_features || 0} onChange={(e) => handleInfoChange('n_cat_features', parseInt(e.target.value) || 0)} />
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
                            <div className="row">
                                {Object.entries(domainData).map(([key, value]) => {
                                    const isNumerical = key.startsWith('num_attr_');
                                    const labelText = columnNameMap[key] || key;

                                    return (
                                        <div className="col-md-6 mb-3 d-flex justify-content-center align-items-center flex-column" key={key}>
                                            <label htmlFor={`domain_${key}`} className="form-label">{labelText}</label>
                                            {isNumerical ? (
                                                <div className="d-flex flex-row align-items-center">
                                                    <label htmlFor={`domain_${key}_min`} className="form-label me-2">Min:</label>
                                                    <input
                                                        type="number"
                                                        id={`domain_${key}_min`}
                                                        className="form-control me-3"
                                                        value={value.min || 0}
                                                        onChange={(e) => handleDomainChange(key, e.target.value, 'min')}
                                                    />
                                                    <label htmlFor={`domain_${key}_max`} className="form-label me-2">Max:</label>
                                                    <input
                                                        type="number"
                                                        id={`domain_${key}_max`}
                                                        className="form-control"
                                                        value={value.max || 0}
                                                        onChange={(e) => handleDomainChange(key, e.target.value, 'max')}
                                                    />
                                                </div>
                                            ) : (
                                                <input
                                                    type="number"
                                                    id={`domain_${key}`}
                                                    className="form-control"
                                                    value={value || 0}
                                                    onChange={(e) => handleDomainChange(key, e.target.value)}
                                                />
                                            )}
                                            {!isNumerical && <small className="text-muted">Max num of categories</small>}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    </div>

                    <div className="d-flex justify-content-end mt-4">
                        <button type="button" className="btn btn-secondary me-2" onClick={onCancel}>Cancel</button>
                        <button type="submit" className="btn btn-primary">Confirm & Synthesize</button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default MetadataConfirmation;

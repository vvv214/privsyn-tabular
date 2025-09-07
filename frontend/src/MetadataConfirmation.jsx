import React, { useState, useEffect } from 'react';

const MetadataConfirmation = ({ uniqueId, inferredDomainData, inferredInfoData, synthesisParams, onConfirm, onCancel }) => {
    const [domainData, setDomainData] = useState(inferredDomainData);
    const [infoData, setInfoData] = useState(inferredInfoData);

    useEffect(() => {
        setDomainData(inferredDomainData);
        setInfoData(inferredInfoData);
    }, [inferredDomainData, inferredInfoData]);

    

    const handleDomainChange = (key, subKey, value) => {
        setDomainData(prev => {
            const updatedDomain = {
                ...prev,
                [key]: {
                    ...prev[key],
                    [subKey]: subKey === 'type' ? value : parseFloat(value) || 0
                }
            };

            // After changing the type, we need to recalculate the counts for n_num_features and n_cat_features
            if (subKey === 'type') {
                let num_count = 0;
                let cat_count = 0;
                Object.values(updatedDomain).forEach(v => {
                    if (v.type === 'numerical') num_count++;
                    else if (v.type === 'categorical') cat_count++;
                });
                setInfoData(prevInfo => ({
                    ...prevInfo,
                    n_num_features: num_count,
                    n_cat_features: cat_count
                }));
            }
            return updatedDomain;
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
                                {Object.entries(domainData).map(([key, value]) => (
                                    <div className="col-md-6 mb-4" key={key}>
                                        <div className="card h-100">
                                            <div className="card-body">
                                                <h5 className="card-title text-center mb-3">{key}</h5>
                                                <div className="mb-2">
                                                    <label htmlFor={`domain_${key}_type`} className="form-label">Type</label>
                                                    <select
                                                        id={`domain_${key}_type`}
                                                        className="form-select"
                                                        value={value.type}
                                                        onChange={(e) => handleDomainChange(key, 'type', e.target.value)}
                                                    >
                                                        <option value="categorical">Categorical</option>
                                                        <option value="numerical">Numerical</option>
                                                    </select>
                                                </div>
                                                <div>
                                                    <label htmlFor={`domain_${key}_size`} className="form-label">{value.type === 'numerical' ? 'Unique Values' : 'Categories'}</label>
                                                    <input
                                                        type="number"
                                                        id={`domain_${key}_size`}
                                                        className="form-control"
                                                        value={value.size || 0}
                                                        onChange={(e) => handleDomainChange(key, 'size', e.target.value)}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                ))}
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

import React, { useState, useEffect } from 'react';

const MetadataConfirmation = ({ uniqueId, inferredDomainData, inferredInfoData, synthesisParams, onConfirm, onCancel }) => {
    const [domainData, setDomainData] = useState(inferredDomainData);
    const [infoData, setInfoData] = useState(inferredInfoData);

    useEffect(() => {
        setDomainData(inferredDomainData);
        setInfoData(inferredInfoData);
    }, [inferredDomainData, inferredInfoData]);

    

    const handleDomainChange = (key, value, subKey) => {
        setDomainData(prevDomainData => {
            const oldType = prevDomainData[key].type;
            const newType = subKey === 'type' ? value : oldType;

            const updatedDomain = {
                ...prevDomainData,
                [key]: {
                    ...prevDomainData[key],
                    [subKey]: subKey === 'size' ? parseFloat(value) || 0 : value
                }
            };

            // If type has changed, update infoData counts
            if (subKey === 'type' && oldType !== newType) {
                setInfoData(prevInfoData => {
                    const n_num_features = prevInfoData.n_num_features + (newType === 'numerical' ? 1 : -1);
                    const n_cat_features = prevInfoData.n_cat_features + (newType === 'categorical' ? 1 : -1);
                    return { ...prevInfoData, n_num_features, n_cat_features };
                });
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
                            <div className="table-responsive">
                                <table className="table table-bordered table-hover">
                                    <thead className="table-light">
                                        <tr>
                                            <th>Column Name</th>
                                            <th>Data Type</th>
                                            <th>Size / Categories</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {Object.entries(domainData).map(([key, value]) => (
                                            <tr key={key}>
                                                <td>{key}</td>
                                                <td>
                                                    <select
                                                        className="form-select"
                                                        value={value.type}
                                                        onChange={(e) => handleDomainChange(key, e.target.value, 'type')}
                                                    >
                                                        <option value="numerical">Numerical</option>
                                                        <option value="categorical">Categorical</option>
                                                    </select>
                                                </td>
                                                <td>
                                                    <input
                                                        type="number"
                                                        className="form-control"
                                                        value={value.size || 0}
                                                        onChange={(e) => handleDomainChange(key, e.target.value, 'size')}
                                                    />
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
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

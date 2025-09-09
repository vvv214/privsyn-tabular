import { useState } from 'react';

function SynthesisForm({
  formData,
  handleChange,
  handleFileChange,
  handleSubmit,
  handleLoadSample,
  loadingSample,
}) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <section className="card">
      <div className="card-header">
        <h3 className="mb-0">Synthesis Parameters</h3>
      </div>
      <div className="card-body">
        <form onSubmit={handleSubmit}>
          {/* Basic Settings */}
          <div className="row mb-3">
            <div className="col-12">
              <label htmlFor="dataset_name" className="form-label">Dataset Name</label>
              <input
                type="text"
                className="form-control"
                id="dataset_name"
                name="dataset_name"
                value={formData.dataset_name}
                onChange={handleChange}
                placeholder="e.g., my_dataset"
                required
              />
            </div>
          </div>

          <div className="row mb-3">
            <div className="col-md-4">
              <label htmlFor="epsilon" className="form-label">Epsilon (ε)</label>
              <input
                type="number"
                step="any"
                className="form-control"
                id="epsilon"
                name="epsilon"
                value={formData.epsilon}
                onChange={handleChange}
                required
              />
            </div>
            <div className="col-md-4">
              <label htmlFor="delta" className="form-label">Delta (δ)</label>
              <input
                type="number"
                step="any"
                className="form-control"
                id="delta"
                name="delta"
                value={formData.delta}
                onChange={handleChange}
                required
              />
            </div>
            <div className="col-md-4">
              <label htmlFor="n_sample" className="form-label">Number of Samples</label>
              <input
                type="number"
                className="form-control"
                id="n_sample"
                name="n_sample"
                value={formData.n_sample}
                onChange={handleChange}
                required
              />
            </div>
          </div>

          {/* File Upload */}
          <h5 className="mt-4 mb-3">Upload Dataset File</h5>
          <div className="mb-3">
            <label htmlFor="data_file" className="form-label">Select CSV or Zip File</label>
            <div className="input-group">
              <input
                type="file"
                className="form-control"
                id="data_file"
                name="data_file"
                onChange={handleFileChange}
                accept=".csv,.zip"
                required={!formData.dataset_name.includes('adult')}
              />
              <button type="button" className="btn btn-outline-secondary" onClick={handleLoadSample} disabled={loadingSample}>
                {loadingSample ? 'Loading...' : 'Load Sample'}
              </button>
            </div>
          </div>

          {/* Advanced Settings Toggle */}
          <div className="d-grid">
            <button
              type="button"
              className="btn btn-link text-secondary text-decoration-none p-0"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
            </button>
          </div>

          {/* Advanced Settings Form */}
          {showAdvanced && (
            <div className="mt-4 p-3 bg-light rounded">
              <h5 className="mb-3">Advanced Parameters</h5>
              <div className="row mb-3">
                <div className="col-md-6">
                  <label htmlFor="method" className="form-label">Method</label>
                  <select
                    className="form-select"
                    id="method"
                    name="method"
                    value={formData.method}
                    onChange={handleChange}
                    required
                  >
                    <option value="privsyn">privsyn</option>
                  </select>
                </div>
                <div className="col-md-6">
                  <label htmlFor="num_preprocess" className="form-label">Numerical Preprocessing</label>
                  <select
                    className="form-select"
                    id="num_preprocess"
                    name="num_preprocess"
                    value={formData.num_preprocess}
                    onChange={handleChange}
                    required
                  >
                    <option value="uniform_kbins">Uniform K-Bins</option>
                    <option value="exp_kbins">Exponential K-Bins</option>
                    <option value="privtree">PrivTree</option>
                    <option value="dawa">DAWA</option>
                    <option value="none">None</option>
                  </select>
                </div>
              </div>
              <div className="row mb-3">
                {formData.method === 'privsyn' && (
                  <div className="col-md-6">
                    <label htmlFor="update_iterations" className="form-label">Update Iterations</label>
                    <input
                      type="number"
                      className="form-control"
                      id="update_iterations"
                      name="update_iterations"
                      value={formData.update_iterations}
                      onChange={handleChange}
                      required
                    />
                    <div className="form-text">More iterations improve quality but are slower.</div>
                  </div>
                )}
              </div>
              <div className="row">
                <div className="col-md-6">
                  <div className="form-check">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id="append"
                      name="append"
                      checked={formData.append}
                      onChange={handleChange}
                    />
                    <label className="form-check-label" htmlFor="append">
                      Append
                    </label>
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="form-check">
                    <input
                      className="form-check-input"
                      type="checkbox"
                      id="sep_syn"
                      name="sep_syn"
                      checked={formData.sep_syn}
                      onChange={handleChange}
                    />
                    <label className="form-check-label" htmlFor="sep_syn">
                      Separate Synthesis
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="d-grid mt-4">
            <button type="submit" className="btn btn-primary btn-lg">
              Infer Metadata & Synthesize
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}

export default SynthesisForm;

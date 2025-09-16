function SynthesisForm({
  formData,
  handleChange,
  handleFileChange,
  handleSubmit,
  handleLoadSample,
  loadingSample,
}) {
  return (
    <section className="card">
      <div className="card-header">
        <h2 className="h5 mb-0">Synthesis Parameters</h2>
      </div>
      <div className="card-body">
        <form onSubmit={handleSubmit}>
          <div className="row mb-4">
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

          <div className="row mb-4">
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

          <div className="mb-4">
            <label htmlFor="data_file" className="form-label">Upload Dataset</label>
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
            <div className="form-text">Upload a CSV or Zip file, or load the sample "adult" dataset.</div>
          </div>

          <div className="mt-3 p-4 bg-light rounded-3">
            <h3 className="h6 mb-3">Advanced Parameters</h3>
            <div className="row g-3">
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
                  <option value="privsyn">PrivSyn</option>
                  <option value="aim">AIM</option>
                </select>
              </div>
              <div className="col-md-6">
                <label className="form-label">Numerical Preprocessing</label>
                <input type="hidden" name="num_preprocess" value={formData.num_preprocess} />
                <div className="form-control-plaintext px-3 py-2 bg-light border rounded">
                  Configure per-column binning after metadata inference.
                </div>
                <div className="form-text">We fall back to uniform bins if no overrides are supplied.</div>
              </div>
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
          </div>

          <hr className="my-4" />

          <div className="d-grid">
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

import { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css'; // Import Bootstrap CSS
import axios from 'axios'; // Import axios
import Papa from 'papaparse'; // Import PapaParse for CSV parsing
import MetadataConfirmation from './MetadataConfirmation'; // Import the new component

const API_URL = import.meta.env.VITE_API_BASE_URL;

function App() {
  const [currentPage, setCurrentPage] = useState('form'); // 'form', 'confirm_metadata', or 'result'
  const [formData, setFormData] = useState({
    method: 'privsyn', // Default value
    dataset_name: '',
    epsilon: 1.0, // Default value
    delta: 1e-5, // Default value
    num_preprocess: 'uniform_kbins', // Default value
    rare_threshold: 0.002, // Default value
    n_sample: 1000, // Default value
    consist_iterations: 5,
    non_negativity: 'N3',
    append: true,
    sep_syn: false,
    initialize_method: 'singleton',
    update_method: 'S5',
    update_rate_method: 'U4',
    update_rate_initial: 1.0,
    update_iterations: 3,
  });

  const [dataFile, setDataFile] = useState(null); // Single file input for CSV
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [synthesizedDataPreview, setSynthesizedDataPreview] = useState([]); // For displaying data preview
  const [synthesizedDataHeaders, setSynthesizedDataHeaders] = useState([]); // For displaying data preview headers
  const [evaluationResults, setEvaluationResults] = useState({}); // To store evaluation results

  // New state for metadata confirmation flow
  const [inferredUniqueId, setInferredUniqueId] = useState(null);
  const [inferredDomainData, setInferredDomainData] = useState(null);
  const [inferredInfoData, setInferredInfoData] = useState(null);

  

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: type === 'checkbox' ? checked : value,
    }));
  };

  const handleFileChange = (e) => {
    setDataFile(e.target.files[0]); // Set the single data file
  };

  const loadSampleData = async () => {
    try {
      const response = await fetch('/adult.csv.zip');
      const blob = await response.blob();
      const file = new File([blob], 'adult.csv.zip', { type: 'application/zip' });
      setDataFile(file);
      setFormData(prev => ({ ...prev, dataset_name: 'adult_sample' }));
      setMessage('Sample dataset (adult.csv.zip) loaded.');
    } catch (error) {
      console.error('Error loading sample data:', error);
      setError('Failed to load sample data.');
    }
  };

  

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('Inferring metadata...');
    setError('');
    setDownloadUrl('');
    setSynthesizedDataPreview([]);
    setSynthesizedDataHeaders([]);
    setEvaluationResults({}); // Clear previous evaluation results

    const data = new FormData();
    for (const key in formData) {
      data.append(key, formData[key]);
    }
    data.append('target_column', 'y_attr'); // Hardcode or infer default

    if (dataFile) {
      data.append('data_file', dataFile); // Append the single data file
    } else {
      setError('Please upload a data file.');
      setMessage('');
      return;
    }

    try {
            const response = await axios.post(`${API_URL}/synthesize`, data, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Store inferred metadata and switch to confirmation page
      setInferredUniqueId(response.data.unique_id);
      setInferredDomainData(response.data.domain_data);
      setInferredInfoData(response.data.info_data);
      setCurrentPage('confirm_metadata');
      setMessage(''); // Clear message after successful inference

    } catch (err) {
      console.error('Synthesis error:', err);
      setError(err.response?.data?.detail || 'Failed to synthesize data. Check console for details.');
      setMessage('');
    }
  };

  const handleConfirmMetadata = async (uniqueId, confirmedDomainData, confirmedInfoData) => {
    setMessage('Initiating data synthesis... This may take a few minutes.');
    setError('');
    setCurrentPage('result'); // Switch to a page that can show progress

    const data = new FormData();
    data.append('unique_id', uniqueId);
    data.append('confirmed_domain_data', JSON.stringify(confirmedDomainData));
    data.append('confirmed_info_data', JSON.stringify(confirmedInfoData));

    for (const key in formData) {
      data.append(key, formData[key]);
    }

    try {
      const response = await axios.post(`${API_URL}/confirm_synthesis`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const { job_id, dataset_name } = response.data;
      setMessage(`Synthesis job started with ID: ${job_id}. Polling for results...`);

      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await axios.get(`${API_URL}/synthesis_status/${job_id}`);
          const { status, error: jobError } = statusResponse.data;

          if (status === 'completed') {
            clearInterval(pollInterval);
            setMessage('Synthesis complete! Fetching results...');
            setDownloadUrl(`${API_URL}/download_synthesized_data/${dataset_name}`);

            const csvResponse = await axios.get(`${API_URL}/download_synthesized_data/${dataset_name}`, { responseType: 'blob' });
            Papa.parse(csvResponse.data, {
              header: true,
              skipEmptyLines: true,
              complete: (results) => {
                setSynthesizedDataHeaders(results.meta.fields || []);
                setSynthesizedDataPreview(results.data.slice(0, 10));
                handleEvaluate(['tvd']);
              },
              error: (err) => {
                console.error("CSV parsing error:", err);
                setError("Failed to parse synthesized data for preview.");
              }
            });
          } else if (status === 'failed') {
            clearInterval(pollInterval);
            setError(`Synthesis failed: ${jobError}`);
            setMessage('');
          } else {
            setMessage(`Synthesis in progress... Status: ${status}`);
          }
        } catch (pollError) {
          clearInterval(pollInterval);
          console.error('Polling error:', pollError);
          setError('Failed to get synthesis status.');
          setMessage('');
        }
      }, 5000); // Poll every 5 seconds

    } catch (err) {
      console.error('Confirmation and Synthesis error:', err);
      setError(err.response?.data?.detail || 'Failed to confirm metadata and synthesize data. Check console for details.');
      setMessage('');
    }
  };

  const handleCancelConfirmation = () => {
    setInferredUniqueId(null);
    setInferredDomainData(null);
    setInferredInfoData(null);
    setCurrentPage('form'); // Go back to the initial form
    setMessage('');
    setError('');
  };

  const handleEvaluate = async (methods) => {
    setMessage('Evaluating data fidelity...');
    setError('');
    setEvaluationResults({}); // Clear previous results

    const data = new FormData();
    data.append('dataset_name', formData.dataset_name);
    data.append('evaluation_methods', methods.map(method => `eval_${method}`).join(','));

    try {
      const response = await axios.post(`${API_URL}/evaluate`, data, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMessage(response.data.message);
      setEvaluationResults(response.data.results);
    } catch (err) {
      console.error('Evaluation error:', err);
      setError(err.response?.data?.detail || 'Failed to run evaluations. Check console for details.');
      setMessage('');
    }
  };

  return (
    <main className="app-container w-100 px-3">
      <div className="mx-auto" style={{ maxWidth: 800, width: '100%' }}>
          
        <h1 className="mb-4 text-center">PrivSyn: A Tool for Differentially Private Data Synthesis</h1>

        <p className="text-center mb-4">
          This tool allows you to synthesize data with differential privacy.
          Upload your dataset, and this tool will infer the metadata, which you can then confirm or correct.
          After that, you can choose the synthesis parameters and generate a new, synthetic dataset.
        </p>

        {message && <div className="alert alert-info mt-4">{message}</div>}
        {error && <div className="alert alert-danger mt-4">Error: {error}</div>}

        {currentPage === 'form' && (
          <section className="card shadow-sm">
            <div className="card-header bg-primary text-white">
              <h3 className="mb-0">Synthesis Parameters</h3>
            </div>
            <div className="card-body">
              <form onSubmit={handleSubmit}>
                {/* Basic Settings */}
                <div className="row mb-3">
                  <div className="col-md-12">
                    <label htmlFor="dataset_name" className="form-label">Dataset Name</label>
                    <input
                      type="text"
                      className="form-control"
                      id="dataset_name"
                      name="dataset_name"
                      value={formData.dataset_name}
                      onChange={handleChange}
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
                    <small className="text-muted">Privacy parameter; smaller is more private.</small>
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
                    <small className="text-muted">Privacy parameter; should be small.</small>
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
                    <small className="text-muted">Number of synthetic records to generate.</small>
                  </div>
                </div>

                {/* File Upload */}
                <h4 className="mt-4 mb-3">Upload Dataset File (CSV or Zip)</h4>
                <div className="mb-3">
                  <label htmlFor="data_file" className="form-label">Select CSV/Zip File</label>
                  <input
                    type="file"
                    className="form-control"
                    id="data_file"
                    name="data_file"
                    onChange={handleFileChange}
                    accept=".csv,.zip"
                  />
                </div>
                <div className="text-center my-3">
                  <p className="mb-2">Or, use a sample dataset:</p>
                  <button
                    type="button"
                    className="btn btn-outline-primary"
                    onClick={loadSampleData}
                  >
                    Load Adult Sample Dataset
                  </button>
                </div>

                {/* Advanced Settings Toggle */}
                <div className="d-grid gap-2">
                  <button
                    type="button"
                    className="btn btn-secondary mt-3"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                  >
                    {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
                  </button>
                </div>

                {/* Advanced Settings Form */}
                {showAdvanced && (
                  <div className="mt-4">
                    <hr />
                    <h4 className="mb-3">Advanced Synthesis Parameters</h4>
                    <div className="row mb-3">
                      <div className="col-md-6">
                        <label htmlFor="method" className="form-label">Method</label>
                        <select
                          className="form-select"
                          id="method"
                          name="method"
                          value={formData.method}
                          onChange={handleChange}
                          disabled
                        >
                          <option value="privsyn">PrivSyn</option>
                        </select>
                        <small className="text-muted">Currently, only the PrivSyn method is supported.</small>
                      </div>
                      <div className="col-md-6">
                        <label htmlFor="num_preprocess" className="form-label">Numerical Preprocess</label>
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
                          <option value="dawa">Dawa</option>
                          <option value="none">None</option>
                        </select>
                      </div>
                    </div>
                    <div className="row mb-3">
                      <div className="col-md-6">
                        <label htmlFor="rare_threshold" className="form-label">Rare Threshold</label>
                        <input
                          type="number"
                          step="any"
                          className="form-control"
                          id="rare_threshold"
                          name="rare_threshold"
                          value={formData.rare_threshold}
                          onChange={handleChange}
                          required
                        />
                        <small className="text-muted">Threshold for merging rare categories.</small>
                      </div>
                      <div className="col-md-6">
                        <label htmlFor="consist_iterations" className="form-label">Consist Iterations</label>
                        <input
                          type="number"
                          className="form-control"
                          id="consist_iterations"
                          name="consist_iterations"
                          value={formData.consist_iterations}
                          onChange={handleChange}
                          required
                        />
                        <small className="text-muted">Number of iterations for marginal consistency.</small>
                      </div>
                    </div>
                    <div className="row mb-3">
                      <div className="col-md-6">
                        <label htmlFor="non_negativity" className="form-label">Non Negativity</label>
                        <input
                          type="text"
                          className="form-control"
                          id="non_negativity"
                          name="non_negativity"
                          value={formData.non_negativity}
                          onChange={handleChange}
                          required
                        />
                      </div>
                      <div className="col-md-6">
                        <label htmlFor="initialize_method" className="form-label">Initialize Method</label>
                        <input
                          type="text"
                          className="form-control"
                          id="initialize_method"
                          name="initialize_method"
                          value={formData.initialize_method}
                          onChange={handleChange}
                          required
                        />
                      </div>
                    </div>
                    <div className="row mb-3">
                      <div className="col-md-6">
                        <label htmlFor="update_method" className="form-label">Update Method</label>
                        <input
                          type="text"
                          className="form-control"
                          id="update_method"
                          name="update_method"
                          value={formData.update_method}
                          onChange={handleChange}
                          required
                        />
                      </div>
                      <div className="col-md-6">
                        <label htmlFor="update_rate_method" className="form-label">Update Rate Method</label>
                        <input
                          type="text"
                          className="form-control"
                          id="update_rate_method"
                          name="update_rate_method"
                          value={formData.update_rate_method}
                          onChange={handleChange}
                          required
                        />
                      </div>
                    </div>
                    <div className="row mb-3">
                      <div className="col-md-6">
                        <label htmlFor="update_rate_initial" className="form-label">Update Rate Initial</label>
                        <input
                          type="number"
                          step="any"
                          className="form-control"
                          id="update_rate_initial"
                          name="update_rate_initial"
                          value={formData.update_rate_initial}
                          onChange={handleChange}
                          required
                        />
                      </div>
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
                      </div>
                    </div>
                    <div className="row mb-3">
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

                <div className="d-grid gap-2">
                  <button type="submit" className="btn btn-primary mt-3">Infer Metadata & Synthesize</button>
                </div>
              </form>
            </div>
          </section>
        )}

        {currentPage === 'confirm_metadata' && inferredDomainData && inferredInfoData && (
          <MetadataConfirmation
            uniqueId={inferredUniqueId}
            inferredDomainData={inferredDomainData}
            inferredInfoData={inferredInfoData}
            synthesisParams={formData}
            onConfirm={handleConfirmMetadata}
            onCancel={handleCancelConfirmation}
          />
        )}

        {currentPage === 'result' && (
          <section className="card shadow-sm mt-5">
            <div className="card-header bg-success text-white text-center">
              <h3 className="mb-0">Synthesis Results for {formData.dataset_name}</h3>
            </div>
            <div className="card-body">
              {!downloadUrl && !error && (
                <div className="text-center">
                  <div className="spinner-border text-primary" role="status">
                    <span className="visually-hidden">Loading...</span>
                  </div>
                  <p className="mt-3">Synthesis in progress. This may take several minutes. Please do not refresh the page.</p>
                </div>
              )}
              {downloadUrl && (
                <div>
                  <div className="text-center">
                    <a href={downloadUrl} download={`${formData.dataset_name}_synthesized.csv`} className="btn btn-primary mb-3">
                      Download Synthesized Data
                    </a>
                  </div>

                  <h4 className="mt-4 mb-3 text-center">Synthesized Data Preview (First 10 Rows)</h4>
                  {synthesizedDataPreview.length > 0 ? (
                    <div className="text-center">
                      <div className="table-responsive">
                        <table className="table table-striped table-bordered table-hover d-inline-block">
                          <thead>
                            <tr>
                              {synthesizedDataHeaders.map((header, index) => (
                                <th key={index}>{header}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {synthesizedDataPreview.map((row, rowIndex) => (
                              <tr key={rowIndex}>
                                {synthesizedDataHeaders.map((header, colIndex) => (
                                  <td key={colIndex}>{row[header]}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <p className="text-muted text-center">No data preview available.</p>
                  )}

                  <hr className="my-4" />

                  {Object.keys(evaluationResults).length > 0 && (
                    <div className="mt-4">
                      <h5 className="mb-3 text-center">Evaluation Results:</h5>
                      {Object.entries(evaluationResults).map(([method, result]) => (
                        <div key={method} className="card mb-3">
                          <div className="card-header bg-light">{method.replace('eval_', '').replace('_', '').toUpperCase()}</div>
                          <div className="card-body">
                            <pre className="card-text small text-start">{JSON.stringify(result, null, 2)}</pre>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>
        )}
      </div>
    </main>
  );
}

export default App;
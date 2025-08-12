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
    n_sample: 5000, // Default value
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

  const evaluationMethods = [
    'eval_catboost', 'eval_mlp', 'eval_query', 'eval_sample',
    'eval_seeds', 'eval_simple', 'eval_transformer', 'eval_tvd'
  ];
  const [selectedEvaluations, setSelectedEvaluations] = useState([]);

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

  const handleEvaluationChange = (e) => {
    const { value, checked } = e.target;
    setSelectedEvaluations((prevSelected) =>
      checked ? [...prevSelected, value] : prevSelected.filter((method) => method !== value)
    );
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
      const response = await axios.post('${API_URL}/synthesize', data, {
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
    setMessage('Synthesizing data...');
    setError('');

    const data = new FormData();
    data.append('unique_id', uniqueId);
    data.append('confirmed_domain_data', JSON.stringify(confirmedDomainData));
    data.append('confirmed_info_data', JSON.stringify(confirmedInfoData));

    // Append all synthesis parameters from formData
    for (const key in formData) {
      data.append(key, formData[key]);
    }

    try {
      const response = await axios.post('${API_URL}/confirm_synthesis', data, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const { message: responseMessage, dataset_name: returnedDatasetName } = response.data;
      setMessage(responseMessage);
      setDownloadUrl(`http://localhost:8001/download_synthesized_data/${returnedDatasetName}`);

      // Fetch the synthesized CSV for preview
      const csvResponse = await axios.get(`http://localhost:8001/download_synthesized_data/${returnedDatasetName}`, { responseType: 'blob' });

      Papa.parse(csvResponse.data, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          setSynthesizedDataHeaders(results.meta.fields || []);
          setSynthesizedDataPreview(results.data.slice(0, 10)); // Show first 10 rows
          setCurrentPage('result'); // Switch to result page
        },
        error: (err) => {
          console.error("CSV parsing error:", err);
          setError("Failed to parse synthesized data for preview.");
          setMessage('');
        }
      });

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

  const handleEvaluate = async () => {
    setMessage('Evaluating data fidelity...');
    setError('');
    setEvaluationResults({}); // Clear previous results

    const data = new FormData();
    data.append('dataset_name', formData.dataset_name);
    data.append('evaluation_methods', selectedEvaluations.join(','));

    try {
      const response = await axios.post('${API_URL}/evaluate', data, {
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
    <div className="app-container w-100 px-3">
      <div className="mx-auto" style={{ maxWidth: 800, width: '100%' }}>
          
        <h1 className="mb-4 text-center">PrivSyn Data Synthesizer</h1>

        {message && <div className="alert alert-info mt-4">{message}</div>}
        {error && <div className="alert alert-danger mt-4">Error: {error}</div>}

        {currentPage === 'form' && (
          <div className="card shadow-sm">
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
                    <label htmlFor="epsilon" className="form-label">Epsilon</label>
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
                    <label htmlFor="delta" className="form-label">Delta</label>
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
                    required
                  />
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
                        <input
                          type="text"
                          className="form-control"
                          id="method"
                          name="method"
                          value={formData.method}
                          onChange={handleChange}
                          required
                        />
                      </div>
                      <div className="col-md-6">
                        <label htmlFor="num_preprocess" className="form-label">Numerical Preprocess</label>
                        <input
                          type="text"
                          className="form-control"
                          id="num_preprocess"
                          name="num_preprocess"
                          value={formData.num_preprocess}
                          onChange={handleChange}
                          required
                        />
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
          </div>
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
          <div className="card shadow-sm mt-5">
            <div className="card-header bg-success text-white text-center">
              <h3 className="mb-0">Synthesis Results for {formData.dataset_name}</h3>
            </div>
            <div className="card-body">
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

              <h4 className="mt-4 mb-3 text-center">Evaluate Data Fidelity</h4>
              <div className="d-flex justify-content-center">
                  <div className="row mb-3">
                  {evaluationMethods.map((method) => (
                      <div className="col-md-4" key={method}>
                      <div className="form-check">
                          <input
                          className="form-check-input"
                          type="checkbox"
                          value={method}
                          id={`eval-${method}`}
                          onChange={handleEvaluationChange}
                          checked={selectedEvaluations.includes(method)}
                          />
                          <label className="form-check-label" htmlFor={`eval-${method}`}>
                            {method.replace('eval_', '').replace('_', ' ').toUpperCase()}
                          </label>
                      </div>
                      </div>
                  ))}
                  </div>
              </div>
              <div className="text-center">
                  <button
                  type="button"
                  className="btn btn-secondary mt-3"
                  onClick={handleEvaluate}
                  disabled={selectedEvaluations.length === 0}
                  >
                  Run Selected Evaluations
                  </button>
              </div>

              {Object.keys(evaluationResults).length > 0 && (
                <div className="mt-4">
                  <h5 className="mb-3 text-center">Evaluation Results:</h5>
                  {Object.entries(evaluationResults).map(([method, result]) => (
                    <div key={method} className="card mb-3">
                      <div className="card-header bg-light">{method.replace('eval_', '').replace('_', '').toUpperCase()}</div>
                      <div className="card-body">
                        <pre className="card-text small text-start">{result}</pre>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
import { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css'; // Import Bootstrap CSS
import axios from 'axios'; // Import axios
import Papa from 'papaparse'; // Import PapaParse for CSV parsing

function App() {
  const [currentPage, setCurrentPage] = useState('form'); // 'form' or 'result'
  const [formData, setFormData] = useState({
    method: 'privsyn', // Default value
    dataset_name: '',
    epsilon: 1.0, // Default value
    delta: 1e-5, // Default value
    num_preprocess: 'uniform_kbins', // Default value
    rare_threshold: 0.002, // Default value
    n_sample: 5000, // Default value
    // target_column is removed from UI, but still sent to backend with default
  });

  const [dataFile, setDataFile] = useState(null); // Single file input for CSV

  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [synthesizedDataPreview, setSynthesizedDataPreview] = useState([]); // For displaying data preview
  const [synthesizedDataHeaders, setSynthesizedDataHeaders] = useState([]); // For displaying data preview headers
  const [evaluationResults, setEvaluationResults] = useState({}); // To store evaluation results

  const evaluationMethods = [
    'eval_catboost', 'eval_mlp', 'eval_query', 'eval_sample',
    'eval_seeds', 'eval_simple', 'eval_transformer', 'eval_tvd'
  ];
  const [selectedEvaluations, setSelectedEvaluations] = useState([]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
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
    setMessage('Synthesizing data...');
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
      const response = await axios.post('http://localhost:8001/synthesize', data, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Expecting JSON response with paths
      const { message: responseMessage, synthesized_csv_path, dataset_name: returnedDatasetName } = response.data;

      setMessage(responseMessage);
      // Set download URL to the new endpoint
      setDownloadUrl(`http://localhost:8001/download_synthesized_data/${returnedDatasetName}`);

      // Fetch the synthesized CSV for preview
      const csvResponse = await axios.get(downloadUrl, { responseType: 'blob' });

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
      console.error('Synthesis error:', err);
      setError(err.response?.data?.detail || 'Failed to synthesize data. Check console for details.');
      setMessage('');
    }
  };

  const handleEvaluate = async () => {
    setMessage('Evaluating data fidelity...');
    setError('');
    setEvaluationResults({}); // Clear previous results

    const data = new FormData();
    data.append('dataset_name', formData.dataset_name);
    data.append('evaluation_methods', selectedEvaluations.join(','));

    try {
      const response = await axios.post('http://localhost:8001/evaluate', data, {
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
    <div className="container mt-5">
      <h1 className="mb-4">PrivSyn Data Synthesizer</h1>

      {currentPage === 'form' && (
        <form onSubmit={handleSubmit}>
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

        <div className="row mb-3">
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
        </div>

        <h3 className="mt-4 mb-3">Upload Dataset File (CSV)</h3>
        <div className="mb-3">
          <label htmlFor="data_file" className="form-label">Select CSV File</label>
          <input
            type="file"
            className="form-control"
            id="data_file"
            name="data_file"
            onChange={handleFileChange}
            accept=".csv"
            required
          />
        </div>

        <button type="submit" className="btn btn-primary mt-3">Synthesize Data</button>
      </form>
      )}

      {currentPage === 'result' && (
        <div className="mt-5">
          <h2>Synthesis Results for {formData.dataset_name}</h2>
          {message && <div className="alert alert-info mt-4">{message}</div>}
          {error && <div className="alert alert-danger mt-4">Error: {error}</div>}
          {downloadUrl && (
            <div className="alert alert-success mt-4">
              <a href={downloadUrl} download={`${formData.dataset_name}_synthesized.csv`}>
                Download Synthesized Data
              </a>
            </div>
          )}

          <h3 className="mt-4">Synthesized Data Preview (First 10 Rows)</h3>
          {synthesizedDataPreview.length > 0 ? (
            <div className="table-responsive">
              <table className="table table-striped table-bordered">
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
          ) : (
            <p>No data preview available.</p>
          )}

          <h3 className="mt-4">Evaluate Data Fidelity</h3>
          <p>Select evaluation methods:</p>
          <div className="row">
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
          <button
            type="button"
            className="btn btn-secondary mt-3"
            onClick={handleEvaluate}
            disabled={selectedEvaluations.length === 0}
          >
            Run Selected Evaluations
          </button>

          {Object.keys(evaluationResults).length > 0 && (
            <div className="mt-4">
              <h4>Evaluation Results:</h4>
              {Object.entries(evaluationResults).map(([method, result]) => (
                <div key={method} className="card mb-3">
                  <div className="card-header">{method.replace('eval_', '').replace('_', ' ').toUpperCase()}</div>
                  <div className="card-body">
                    <pre className="card-text">{result}</pre>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;

import { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css'; // Import Bootstrap CSS
import axios from 'axios'; // Import axios

function App() {
  const [formData, setFormData] = useState({
    method: 'privsyn', // Default value
    dataset_name: '',
    epsilon: 1.0, // Default value
    delta: 1e-5, // Default value
    num_preprocess: 'uniform_kbins', // Default value
    rare_threshold: 0.002, // Default value
    n_sample: 5000, // Default value
    target_column: 'y_attr', // New form field for target column
  });

  const [dataFile, setDataFile] = useState(null); // Single file input for CSV

  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('Synthesizing data...');
    setError('');
    setDownloadUrl('');

    const data = new FormData();
    for (const key in formData) {
      data.append(key, formData[key]);
    }
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
        responseType: 'blob', // Important for handling file downloads
      });

      if (response.data.type === 'application/json') {
        // If the response is JSON, it's likely an error
        const errorData = JSON.parse(await response.data.text());
        setError(errorData.error || 'An unknown error occurred.');
        setMessage('');
      } else {
        // Assume it's a CSV file
        const url = window.URL.createObjectURL(new Blob([response.data]));
        setDownloadUrl(url);
        setMessage('Synthesis complete! Click the link to download.');
      }
    } catch (err) {
      console.error('Synthesis error:', err);
      setError(err.response?.data?.error || 'Failed to synthesize data. Check console for details.');
      setMessage('');
    }
  };

  return (
    <div className="container mt-5">
      <h1 className="mb-4">PrivSyn Data Synthesizer</h1>
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
        <div className="mb-3">
          <label htmlFor="target_column" className="form-label">Target Column Name (e.g., y_attr)</label>
          <input
            type="text"
            className="form-control"
            id="target_column"
            name="target_column"
            value={formData.target_column}
            onChange={handleChange}
          />
        </div>

        <button type="submit" className="btn btn-primary mt-3">Synthesize Data</button>
      </form>

      {message && <div className="alert alert-info mt-4">{message}</div>}
      {error && <div className="alert alert-danger mt-4">Error: {error}</div>}
      {downloadUrl && (
        <div className="alert alert-success mt-4">
          <a href={downloadUrl} download={`${formData.dataset_name}_synthesized.csv`}>
            Download Synthesized Data
          </a>
        </div>
      )}
    </div>
  );
}

export default App;

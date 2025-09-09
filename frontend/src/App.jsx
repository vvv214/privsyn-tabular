import { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';
import Papa from 'papaparse';
import MetadataConfirmation from './components/MetadataConfirmation';
import SynthesisForm from './components/SynthesisForm';
import ResultsDisplay from './components/ResultsDisplay';
import './index.css';

const API_URL = import.meta.env.VITE_API_BASE_URL;

function App() {
  const [currentPage, setCurrentPage] = useState('form'); // 'form', 'confirm_metadata', or 'result'
  const [formData, setFormData] = useState({
    method: 'privsyn',
    dataset_name: '',
    epsilon: 1.0,
    delta: 1e-5,
    num_preprocess: 'uniform_kbins',
    rare_threshold: 0.002,
    n_sample: 200,
    consist_iterations: 3,
    append: true,
    sep_syn: false,
    initialize_method: 'singleton',
    update_iterations: 2,
  });

  const [dataFile, setDataFile] = useState(null);
  const [loadingSample, setLoadingSample] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [synthesizedDataPreview, setSynthesizedDataPreview] = useState([]);
  const [synthesizedDataHeaders, setSynthesizedDataHeaders] = useState([]);
  const [evaluationResults, setEvaluationResults] = useState({});
  const [inferredUniqueId, setInferredUniqueId] = useState(null);
  const [inferredDomainData, setInferredDomainData] = useState(null);
  const [inferredInfoData, setInferredInfoData] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({ ...prev, [name]: type === 'checkbox' ? checked : value }));
  };

  const handleFileChange = (e) => {
    setDataFile(e.target.files[0]);
  };

  const handleLoadSample = () => {
    setLoadingSample(true);
    setFormData((prev) => ({ ...prev, dataset_name: 'adult' }));
    setDataFile(null);
    setMessage('Sample dataset "adult" loaded. Adjust parameters and click Synthesize.');
    setError('');
    setLoadingSample(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('Inferring metadata...');
    setError('');
    setDownloadUrl('');
    setSynthesizedDataPreview([]);
    setSynthesizedDataHeaders([]);
    setEvaluationResults({});

    const data = new FormData();
    Object.entries(formData).forEach(([key, value]) => data.append(key, value));
    data.append('target_column', 'y_attr');
    if (dataFile) data.append('data_file', dataFile);

    if (!dataFile && !formData.dataset_name.includes('adult')) {
      setError('Please upload a data file or load the sample dataset.');
      setMessage('');
      return;
    }

    try {
      const response = await axios.post(`${API_URL}/synthesize`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setInferredUniqueId(response.data.unique_id);
      setInferredDomainData(response.data.domain_data);
      setInferredInfoData(response.data.info_data);
      setCurrentPage('confirm_metadata');
      setMessage('');
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail) || 'Failed to synthesize data.');
      setMessage('');
    }
  };

  const handleConfirmMetadata = async (uniqueId, domainData, infoData) => {
    setMessage('Synthesizing data...');
    setError('');

    const data = new FormData();
    data.append('unique_id', uniqueId);
    data.append('confirmed_domain_data', JSON.stringify(domainData));
    data.append('confirmed_info_data', JSON.stringify(infoData));
    Object.entries(formData).forEach(([key, value]) => data.append(key, value));

    try {
      const response = await axios.post(`${API_URL}/confirm_synthesis`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const { message: msg, dataset_name: name } = response.data;
      setMessage(msg);
      setDownloadUrl(`${API_URL}/download_synthesized_data/${name}`);

      const csvResponse = await axios.get(`${API_URL}/download_synthesized_data/${name}`, { responseType: 'blob' });
      Papa.parse(csvResponse.data, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          setSynthesizedDataHeaders(results.meta.fields || []);
          setSynthesizedDataPreview(results.data.slice(0, 10));
          setCurrentPage('result');
          handleEvaluate(['query', 'tvd']);
        },
        error: () => setError('Failed to parse synthesized data for preview.'),
      });
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail) || 'Failed to confirm metadata and synthesize data.');
      setMessage('');
    }
  };

  const handleCancelConfirmation = () => {
    setCurrentPage('form');
    setMessage('');
    setError('');
  };

  const handleEvaluate = async (methods) => {
    setMessage('Evaluating data fidelity...');
    setError('');
    setEvaluationResults({});

    const data = new FormData();
    data.append('dataset_name', formData.dataset_name);
    data.append('evaluation_methods', methods.map((m) => `eval_${m}`).join(','));

    try {
      const response = await axios.post(`${API_URL}/evaluate`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMessage(response.data.message);
      setEvaluationResults(response.data.results);
    } catch (err) {
      const detail = err.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail) || 'Failed to run evaluations.');
      setMessage('');
    }
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'form':
        return (
          <SynthesisForm
            formData={formData}
            handleChange={handleChange}
            handleFileChange={handleFileChange}
            handleSubmit={handleSubmit}
            handleLoadSample={handleLoadSample}
            loadingSample={loadingSample}
          />
        );
      case 'confirm_metadata':
        return (
          <MetadataConfirmation
            uniqueId={inferredUniqueId}
            inferredDomainData={inferredDomainData}
            inferredInfoData={inferredInfoData}
            synthesisParams={formData}
            onConfirm={handleConfirmMetadata}
            onCancel={handleCancelConfirmation}
          />
        );
      case 'result':
        return (
          <ResultsDisplay
            formData={formData}
            downloadUrl={downloadUrl}
            synthesizedDataPreview={synthesizedDataPreview}
            synthesizedDataHeaders={synthesizedDataHeaders}
            evaluationResults={evaluationResults}
          />
        );
      default:
        return null;
    }
  };

  return (
    <main className="app-container container-fluid">
      <div className="mx-auto" style={{ maxWidth: 960 }}>
        <header className="text-center mb-4">
          <h1 className="display-5">PrivSyn</h1>
          <p className="lead text-muted">A Tool for Differentially Private Data Synthesis</p>
        </header>

        {message && <div className="alert alert-info">{message}</div>}
        {error && <div className="alert alert-danger">Error: {error}</div>}

        {renderPage()}
      </div>
    </main>
  );
}

export default App;

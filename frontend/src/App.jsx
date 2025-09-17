import { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';
import Papa from 'papaparse';
import MetadataConfirmation from './components/MetadataConfirmation';
import SynthesisForm from './components/SynthesisForm';
import ResultsDisplay from './components/ResultsDisplay';
import LoadingOverlay from './components/LoadingOverlay';
import './index.css';

const API_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001';

function App() {
  const [currentPage, setCurrentPage] = useState('form'); // 'form', 'confirm_metadata', or 'result'
  const [loading, setLoading] = useState(false);
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
  const [sessionId, setSessionId] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState('');
  const [inferredDomainData, setInferredDomainData] = useState(null);
  const [inferredInfoData, setInferredInfoData] = useState(null);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({ ...prev, [name]: type === 'checkbox' ? checked : value }));
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) {
      setDataFile(null);
      return;
    }

    // Reset previous errors and messages
    setError('');
    setMessage('');

    // Basic file type check for immediate feedback
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.zip')) {
      setError('Invalid file type. Please upload a CSV or a Zip file containing a CSV.');
      setDataFile(null);
      e.target.value = null; // Reset file input
      return;
    }

    // For CSV files, perform validation
    if (file.name.endsWith('.csv')) {
      Papa.parse(file, {
        skipEmptyLines: true,
        complete: (results) => {
          if (results.errors.length > 0) {
            setError(`Error parsing CSV file: ${results.errors.map((err) => err.message).join(', ')}`);
            setDataFile(null);
            e.target.value = null;
            return;
          }

          if (results.data.length < 2) { // At least one header row and one data row
            setError('CSV file must have a header and at least one row of data.');
            setDataFile(null);
            e.target.value = null;
            return;
          }

          const headerLength = results.data[0].length;
          for (let i = 1; i < results.data.length; i++) {
            if (results.data[i].length !== headerLength) {
              setError(`Row ${i + 1} has an inconsistent number of columns. Expected ${headerLength}, found ${results.data[i].length}.`);
              setDataFile(null);
              e.target.value = null;
              return;
            }
          }

          // If all checks pass
          setDataFile(file);
          setMessage('File is valid and ready for processing.');
        },
        error: (err) => {
          setError(`An unexpected error occurred while parsing the file: ${err.message}`);
          setDataFile(null);
          e.target.value = null;
        },
      });
    } else {
      // For zip files, we'll rely on backend validation for now
      setDataFile(file);
      setMessage('Zip file selected. Validation will occur on the server.');
    }
  };

  const handleLoadSample = () => {
    setLoadingSample(true);
    setFormData((prev) => ({ ...prev, dataset_name: 'adult' }));
    setDataFile(null);
    setMessage('Sample dataset "adult" loaded. Adjust parameters and click Synthesize.');
    setError('');
    setLoadingSample(false);
  };

  useEffect(() => {
    if (loading) {
      window.scrollTo(0, 0);
    }
  }, [loading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage('Inferring metadata...');
    setError('');
    setDownloadUrl('');
    setSynthesizedDataPreview([]);
    setSynthesizedDataHeaders([]);
    setEvaluationResults({});
    setSessionId(null);
    setEvaluationError('');
    setIsEvaluating(false);

    const data = new FormData();
    Object.entries(formData).forEach(([key, value]) => data.append(key, value));
    data.append('target_column', 'y_attr');
    if (dataFile) data.append('data_file', dataFile);

    if (!dataFile && !formData.dataset_name.includes('adult')) {
      setError('Please upload a data file or load the sample dataset.');
      setMessage('');
      setLoading(false);
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
    } finally {
      setLoading(false);
    }
  };

  const handleConfirmMetadata = async (uniqueId, domainData, infoData) => {
    setLoading(true);
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
      const { message: msg, dataset_name: name, session_id: newSessionId } = response.data;
      setMessage(msg);
      setSessionId(newSessionId);
      setEvaluationError('');
      setIsEvaluating(false);
      const downloadEndpoint = `${API_URL}/download_synthesized_data/${newSessionId}`;
      setDownloadUrl(downloadEndpoint);

      const csvResponse = await axios.get(downloadEndpoint, { responseType: 'blob' });
      Papa.parse(csvResponse.data, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          setSynthesizedDataHeaders(results.meta.fields || []);
          setSynthesizedDataPreview(results.data.slice(0, 10));
          setCurrentPage('result');
          handleEvaluate(newSessionId);
        },
        error: () => setError('Failed to parse synthesized data for preview.'),
      });
    } catch (err) {
      const detail = err.response?.data?.detail;
      const parsedDetail = typeof detail === 'string' ? detail : detail?.message || JSON.stringify(detail) || 'Failed to confirm metadata and synthesize data.';
      setError(parsedDetail);
      setEvaluationError(parsedDetail);
      setIsEvaluating(false);
      setMessage('');
    } finally {
      setLoading(false);
    }
  };

  const handleCancelConfirmation = () => {
    setCurrentPage('form');
    setMessage('');
    setError('');
    setSessionId(null);
    setEvaluationError('');
    setIsEvaluating(false);
  };

  const parseErrorDetail = (detail) => {
    if (typeof detail === 'string') {
      return detail;
    }
    if (detail?.error && detail?.message) {
      return `${detail.error}: ${detail.message}`;
    }
    if (detail?.message) {
      return detail.message;
    }
    return JSON.stringify(detail) || 'An unknown error occurred.';
  };

  const handleEvaluate = async (overrideSessionId = sessionId) => {
    setMessage('Evaluating data fidelity...');
    setError('');
    setEvaluationError('');
    setEvaluationResults({});

    if (!overrideSessionId) {
      const missingMsg = 'Missing synthesis session identifier. Please rerun the synthesis flow.';
      setError(missingMsg);
      setEvaluationError(missingMsg);
      setMessage('');
      return;
    }

    setIsEvaluating(true);

    const data = new FormData();
    data.append('session_id', overrideSessionId);

    try {
      const response = await axios.post(`${API_URL}/evaluate`, data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setMessage(response.data.message || 'Evaluation complete.');
      setEvaluationResults(response.data.results || {});
      setEvaluationError('');
    } catch (err) {
      const detail = parseErrorDetail(err.response?.data?.detail);
      setError(detail);
      setEvaluationError(detail);
      setMessage('');
    } finally {
      setIsEvaluating(false);
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
            isSubmitting={loading}
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
            isSubmitting={loading}
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
            sessionId={sessionId}
            isEvaluating={isEvaluating}
            evaluationError={evaluationError}
            onRetryEvaluate={() => handleEvaluate(sessionId)}
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

        {loading && <LoadingOverlay message={message} />}
        {message && !loading && <div className="alert alert-info">{message}</div>}
        {error && <div className="alert alert-danger">Error: {error}</div>}

        {renderPage()}
      </div>
    </main>
  );
}

export default App;

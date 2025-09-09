import './LoadingOverlay.css';

function LoadingOverlay({ message }) {
  return (
    <div className="loading-overlay">
      <div className="loading-spinner"></div>
      <p className="loading-message">{message}</p>
    </div>
  );
}

export default LoadingOverlay;

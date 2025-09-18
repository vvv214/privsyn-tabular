import React from 'react';

function ResultsDisplay({
  formData,
  downloadUrl,
  synthesizedDataPreview,
  synthesizedDataHeaders,
  evaluationResults,
  isEvaluating = false,
}) {
  return (
    <section className="card mt-4">
      <div className="card-header">
        <div className="d-flex justify-content-between align-items-center">
          <h2 className="h5 mb-0">Synthesis Results: {formData.dataset_name}</h2>
        </div>
      </div>
      <div className="card-body">
        <div className="text-center mb-4 d-flex flex-column gap-3 align-items-center">
          <a
            href={downloadUrl}
            download={`${formData.dataset_name}_synthesized.csv`}
            className="btn btn-success btn-lg"
            aria-disabled={isEvaluating}
            aria-label="Download Synthesized Data"
            data-testid="download-synth-link"
          >
            {isEvaluating ? 'Download (disabled during evaluation)' : 'Download Synthesized Data'}
          </a>
          {isEvaluating && (
            <div className="text-muted">Evaluating data fidelity…</div>
          )}
        </div>

        <h3 className="h6 text-center mb-3">Synthesized Data Preview (First 10 Rows)</h3>
        {synthesizedDataPreview.length > 0 ? (
          <div className="table-responsive">
            <table className="table table-striped table-bordered table-hover">
              <thead className="table-light">
                <tr>
                  {synthesizedDataHeaders.map((header, index) => (
                    <th key={index} scope="col">{header}</th>
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
          <p className="text-muted text-center">No data preview available.</p>
        )}

        {Object.keys(evaluationResults).length > 0 && (
          <div className="mt-5">
            <hr />
            <h3 className="h6 text-center mb-3">Evaluation Results</h3>
            {Object.entries(evaluationResults).map(([method, result]) => (
              <div key={method} className="card mb-3">
                <div className="card-header bg-light">
                  {method.replace('eval_', '').replace('_', ' ').toUpperCase()}
                </div>
                <div className="card-body">
                  <pre>{JSON.stringify(result, null, 2)}</pre>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

export default ResultsDisplay;

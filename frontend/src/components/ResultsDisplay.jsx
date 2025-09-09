import React from 'react';

function ResultsDisplay({
  formData,
  downloadUrl,
  synthesizedDataPreview,
  synthesizedDataHeaders,
  evaluationResults,
}) {
  return (
    <section className="card mt-4">
      <div className="card-header">
        <h3 className="mb-0">Synthesis Results: {formData.dataset_name}</h3>
      </div>
      <div className="card-body">
        <div className="text-center">
          <a href={downloadUrl} download={`${formData.dataset_name}_synthesized.csv`} className="btn btn-success btn-lg mb-3">
            Download Synthesized Data
          </a>
        </div>

        <h5 className="mt-4 mb-3 text-center">Synthesized Data Preview (First 10 Rows)</h5>
        {synthesizedDataPreview.length > 0 ? (
          <div className="table-responsive">
            <table className="table table-striped table-bordered table-hover">
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
          <p className="text-muted text-center">No data preview available.</p>
        )}

        {Object.keys(evaluationResults).length > 0 && (
          <div className="mt-4">
            <hr />
            <h5 className="mb-3 text-center">Evaluation Results</h5>
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

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
        <h2 className="h5 mb-0">Synthesis Results: {formData.dataset_name}</h2>
      </div>
      <div className="card-body">
        <div className="text-center mb-4">
          <a href={downloadUrl} download={`${formData.dataset_name}_synthesized.csv`} className="btn btn-success btn-lg">
            Download Synthesized Data
          </a>
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

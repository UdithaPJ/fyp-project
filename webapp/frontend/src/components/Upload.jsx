import { useState } from "react";
import { uploadDataset } from "../services/api";

function Upload({
  networkType,
  onNetworkTypeChange,
  uploadResult,
  isDetecting,
  onUploadSuccess,
  onContinue,
}) {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [localError, setLocalError] = useState("");

  async function handleSubmit(event) {
    event.preventDefault();

    if (!file) {
      setLocalError("Choose a file before uploading.");
      return;
    }

    setIsUploading(true);
    setLocalError("");

    try {
      const response = await uploadDataset(file);
      onUploadSuccess(response);
    } catch (error) {
      setLocalError(error.message);
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Upload Dataset</h2>
        <p>
          Select a CSV, TSV, Excel, or JSON file from your machine, then review
          the preview before continuing.
        </p>
      </div>

      <div className="mapping-grid">
        <label>
          Network type
          <select
            onChange={(event) => onNetworkTypeChange?.(event.target.value)}
            value={networkType || "grn"}
          >
            <option value="grn">GRN (directed)</option>
            <option value="ppi">PPI (undirected)</option>
            <option value="mirna">miRNA → target (directed bipartite)</option>
          </select>
        </label>
      </div>

      <form className="upload-form" onSubmit={handleSubmit}>
        <input
          accept=".csv,.tsv,.xlsx,.json,.txt,.tab"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
          type="file"
        />
        <button className="primary-button" disabled={isUploading} type="submit">
          {isUploading ? "Uploading..." : "Upload"}
        </button>
      </form>

      {file ? (
        <p className="helper-text">
          Selected: <strong>{file.name}</strong>
        </p>
      ) : null}

      {localError ? <p className="inline-error">{localError}</p> : null}

      {uploadResult ? (
        <div className="report-box">
          <h3>Preview</h3>
          <p>
            Columns: <strong>{uploadResult.columns.length}</strong>
          </p>
          <p>
            Rows: <strong>{uploadResult.row_count}</strong>
          </p>
          <div className="preview-table-wrap">
            <table className="preview-table">
              <thead>
                <tr>
                  {uploadResult.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {uploadResult.preview.map((row, index) => (
                  <tr key={`row-${index}`}>
                    {uploadResult.columns.map((column) => (
                      <td key={`${index}-${column}`}>{String(row[column] ?? "")}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="action-row">
            <button
              className="primary-button"
              disabled={isDetecting || !uploadResult}
              onClick={onContinue}
              type="button"
            >
              {isDetecting ? "Detecting columns..." : "Continue to Mapping"}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default Upload;

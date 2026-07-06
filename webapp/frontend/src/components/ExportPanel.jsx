import { useMemo, useState } from "react";
import { exportResultsCSV, exportResultsJSON } from "../services/api";

/**
 * Step 7 — Export results as CSV or JSON.
 *
 * Two download buttons trigger browser downloads via the helpers in
 * api.js (which call fetch + create an anchor element).  Two reset
 * buttons let the user run another algorithm or start the whole flow
 * over.
 */

function timestampSuffix() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return (
    now.getFullYear().toString() +
    pad(now.getMonth() + 1) +
    pad(now.getDate()) +
    "_" +
    pad(now.getHours()) +
    pad(now.getMinutes()) +
    pad(now.getSeconds())
  );
}

function ExportPanel({ jobId, algorithmName, onRunAnother, onStartOver }) {
  const [busy, setBusy]   = useState(null); // 'csv' | 'json' | null
  const [error, setError] = useState("");

  const filenameStem = useMemo(
    () => `${algorithmName || "result"}_results_${timestampSuffix()}`,
    [algorithmName],
  );

  async function handleExport(kind) {
    if (!jobId) return;
    setBusy(kind);
    setError("");
    try {
      if (kind === "csv") await exportResultsCSV(jobId);
      else if (kind === "json") await exportResultsJSON(jobId);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Export Results</h2>
        <p>
          Download the analysis output locally — no data leaves your machine.
        </p>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="export-panel">
        <div className="export-card">
          <h3>CSV Export</h3>
          <p>
            Top-nodes or cluster table as a comma-separated value file.
          </p>
          <code className="export-filename">{filenameStem}.csv</code>
          <button
            className="primary-button"
            disabled={busy !== null || !jobId}
            onClick={() => handleExport("csv")}
            type="button"
          >
            {busy === "csv" ? "Preparing…" : "Download CSV"}
          </button>
        </div>

        <div className="export-card">
          <h3>JSON Export</h3>
          <p>
            Full result payload including chart, table, and graph
            visualisation data.
          </p>
          <code className="export-filename">{filenameStem}.json</code>
          <button
            className="primary-button"
            disabled={busy !== null || !jobId}
            onClick={() => handleExport("json")}
            type="button"
          >
            {busy === "json" ? "Preparing…" : "Download JSON"}
          </button>
        </div>
      </div>

      <div className="action-row">
        <button
          className="secondary-button"
          onClick={onStartOver}
          type="button"
        >
          Start Over
        </button>
        <button
          className="primary-button"
          onClick={onRunAnother}
          type="button"
        >
          Run Another Analysis
        </button>
      </div>
    </div>
  );
}

export default ExportPanel;

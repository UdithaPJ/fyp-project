function Validation({
  rowCount,
  duplicateMethod,
  onDuplicateMethodChange,
  onBack,
  onRunPreprocess,
  onContinue,
  validation,
  isPreprocessing,
  hasValidMapping,
  progress,
}) {
  const missingValues = validation?.missing_values || {};
  const progressPercent = Math.max(0, Math.min(100, Math.round((progress?.percentage || 0) * 100)));
  const progressLabel = progress?.step
    ? progress.step.replaceAll("_", " ")
    : "waiting";

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Validation</h2>
        <p>Choose how duplicate edges should be handled and review the results.</p>
      </div>

      <div className="mapping-grid">
        <label>
          <span>Duplicate handling</span>
          <select
            onChange={(event) => onDuplicateMethodChange(event.target.value)}
            value={duplicateMethod}
          >
            <option value="count">count</option>
            <option value="mean">mean</option>
            <option value="max">max</option>
            <option value="none">none</option>
          </select>
        </label>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span>Uploaded rows</span>
          <strong>{rowCount}</strong>
        </div>
        <div className="stat-card">
          <span>Duplicate edges</span>
          <strong>{validation?.duplicate_edges ?? "-"}</strong>
        </div>
        <div className="stat-card">
          <span>Self-loops</span>
          <strong>{validation?.self_loops ?? "-"}</strong>
        </div>
      </div>

      <div className="report-box">
        <h3>Missing values</h3>
        <p>Source: {missingValues.source ?? "-"}</p>
        <p>Target: {missingValues.target ?? "-"}</p>
        <p>Weight: {missingValues.weight ?? "-"}</p>
      </div>

      {isPreprocessing ? (
        <div className="report-box progress-box">
          <div className="progress-header">
            <h3>Preprocessing Progress</h3>
            <strong>{progressPercent}%</strong>
          </div>
          <p className="progress-copy">
            Current step: <span>{progressLabel}</span>
          </p>
          <div aria-hidden="true" className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      ) : null}

      {validation ? (
        <div className="report-box">
          <h3>Validation Report</h3>
          <pre>{JSON.stringify(validation, null, 2)}</pre>
        </div>
      ) : null}

      <div className="action-row">
        <button className="secondary-button" onClick={onBack} type="button">
          Back
        </button>
        {!validation ? (
          <button
            className="primary-button"
            disabled={!hasValidMapping || isPreprocessing}
            onClick={onRunPreprocess}
            type="button"
          >
            {isPreprocessing ? "Running..." : "Validate and Build Graph"}
          </button>
        ) : (
          <button className="primary-button" onClick={onContinue} type="button">
            Confirm Preprocessing
          </button>
        )}
      </div>
    </div>
  );
}

export default Validation;

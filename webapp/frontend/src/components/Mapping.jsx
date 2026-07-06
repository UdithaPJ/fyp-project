function Mapping({
  columns,
  detection,
  mapping,
  onChange,
  onBack,
  onNext,
  isDetecting,
}) {
  function updateField(field, value) {
    onChange({
      ...mapping,
      [field]: value,
    });
  }

  const canContinue = Boolean(mapping.source && mapping.target);

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Column Mapping</h2>
        <p>Review the detected mapping and override any field if needed.</p>
      </div>

      <div className="suggestion-box">
        <p>
          Heuristic suggestion confidence:{" "}
          <strong>{detection ? detection.confidence : 0}</strong>
        </p>
        {isDetecting ? <p>Analyzing sample columns...</p> : null}
      </div>

      <div className="mapping-grid">
        <label>
          <span>Source column</span>
          <select
            onChange={(event) => updateField("source", event.target.value)}
            value={mapping.source}
          >
            <option value="">Select source</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Target column</span>
          <select
            onChange={(event) => updateField("target", event.target.value)}
            value={mapping.target}
          >
            <option value="">Select target</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>

        <label>
          <span>Weight column</span>
          <select
            onChange={(event) => updateField("weight", event.target.value)}
            value={mapping.weight}
          >
            <option value="">No weight column</option>
            {columns.map((column) => (
              <option key={column} value={column}>
                {column}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="action-row">
        <button className="secondary-button" onClick={onBack} type="button">
          Back
        </button>
        <button
          className="primary-button"
          disabled={!canContinue}
          onClick={onNext}
          type="button"
        >
          Continue
        </button>
      </div>
    </div>
  );
}

export default Mapping;

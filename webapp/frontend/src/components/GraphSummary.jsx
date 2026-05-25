import { useState } from "react";
import GraphView from "./GraphView";

function GraphSummary({ preprocessResult, onBack }) {
  const graph = preprocessResult?.graph;
  const [showRawData, setShowRawData] = useState(false);

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Graph Summary</h2>
        <p>The graph has been built and is ready for downstream analysis.</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span>Nodes</span>
          <strong>{preprocessResult?.nodes ?? 0}</strong>
        </div>
        <div className="stat-card">
          <span>Edges</span>
          <strong>{preprocessResult?.edges ?? 0}</strong>
        </div>
      </div>

      <div className="report-box">
        <GraphView graph={graph} />
      </div>

      <div className="report-box">
        <div className="raw-data-header">
          <div>
            <h3>Advanced View</h3>
            <p>Inspect the raw preview payload returned by the backend.</p>
          </div>
          <button
            className="secondary-button"
            onClick={() => setShowRawData((current) => !current)}
            type="button"
          >
            {showRawData ? "Hide Raw Data" : "Show Raw Data"}
          </button>
        </div>

        {showRawData ? (
          <pre>{JSON.stringify(graph || { nodes: {}, edges: [] }, null, 2)}</pre>
        ) : null}
      </div>

      <div className="action-row">
        <button className="secondary-button" onClick={onBack} type="button">
          Back
        </button>
        <button className="primary-button" type="button">
          Run Analysis
        </button>
      </div>
    </div>
  );
}

export default GraphSummary;

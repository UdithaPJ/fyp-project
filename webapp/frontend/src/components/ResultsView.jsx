import { useEffect, useMemo, useState } from "react";
import { getResults } from "../services/api";
import ChartView from "./viz/ChartView";
import GraphHighlight from "./viz/GraphHighlight";
import TableView from "./viz/TableView";

/**
 * Step 6 — Results container with table / chart / graph toggle.
 *
 * The default tab is chosen based on the algorithm type:
 *   pagerank, rwr, hits → Chart
 *   louvain, mcl        → Graph
 *   bfs                 → Table
 *
 * Props
 * -----
 * jobId : string
 * result : dict
 *   The raw algorithm result returned by the runner (already in state from
 *   RunAnalysis).  We additionally fetch the full results payload so we
 *   have chart_data / table_data / graph_viz alongside it.
 * algorithmName : string
 * onContinue : () => void
 *   Forward to the export step.
 */

const TABS = [
  { id: "table", label: "Table" },
  { id: "chart", label: "Chart" },
  { id: "graph", label: "Graph" },
];

function defaultTabForAlgo(algo) {
  if (algo === "louvain" || algo === "mcl") return "graph";
  if (algo === "bfs") return "table";
  return "chart";
}

function ResultsView({ jobId, result, algorithmName, onBack, onContinue }) {
  const [payload, setPayload] = useState(null);
  const [isLoading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState(
    defaultTabForAlgo(algorithmName),
  );

  // Fetch the full payload (result + chart_data + table_data + graph_viz)
  useEffect(() => {
    if (!jobId) return undefined;
    let cancelled = false;
    setLoading(true);
    setError("");
    getResults(jobId)
      .then((data) => {
        if (!cancelled) setPayload(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const executionTime = useMemo(() => {
    return (
      payload?.result?.execution_time ??
      result?.execution_time ??
      null
    );
  }, [payload, result]);

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Results — {algorithmName}</h2>
        <p>
          {payload?.result?.num_nodes ?? result?.num_nodes ?? "-"} nodes,{" "}
          {payload?.result?.num_edges ?? result?.num_edges ?? "-"} edges processed.
          {executionTime != null ? (
            <span className="execution-badge">
              {" "}
              {Number(executionTime).toFixed(4)}s
            </span>
          ) : null}
        </p>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}
      {isLoading ? (
        <div className="report-box">Loading results…</div>
      ) : null}

      <div className="viz-toggle-bar">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`viz-toggle-btn${activeTab === tab.id ? " is-active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
            type="button"
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="viz-content">
        {activeTab === "table" ? (
          <TableView
            algorithmName={algorithmName}
            tableData={payload?.table_data || []}
          />
        ) : null}
        {activeTab === "chart" ? (
          <ChartView chartData={payload?.chart_data || null} />
        ) : null}
        {activeTab === "graph" ? (
          <GraphHighlight graphData={payload?.graph_viz || null} />
        ) : null}
      </div>

      <div className="action-row">
        <button className="secondary-button" onClick={onBack} type="button">
          Back
        </button>
        <button
          className="primary-button"
          disabled={!payload}
          onClick={onContinue}
          type="button"
        >
          Export Results →
        </button>
      </div>
    </div>
  );
}

export default ResultsView;

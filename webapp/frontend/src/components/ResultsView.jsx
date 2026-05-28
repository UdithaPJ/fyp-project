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

const TOP_K_SUPPORTED = new Set(["pagerank", "rwr", "hits"]);
const TOP_K_OPTIONS = [5, 10, 15, 20, 50, 100, 200];

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
  const [topK, setTopK] = useState(null);

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

  // Initialize Top-k once per job (default 10).
  useEffect(() => {
    if (!payload?.job_id) return;
    if (topK != null) return;
    const fromJobParams = payload?.params?.top_k;
    const parsed = Number.parseInt(String(fromJobParams ?? 10), 10);
    setTopK(Number.isFinite(parsed) && parsed > 0 ? parsed : 10);
  }, [payload?.job_id, payload?.params, topK]);

  const executionTime = useMemo(() => {
    return (
      payload?.result?.execution_time ??
      result?.execution_time ??
      null
    );
  }, [payload, result]);

  const metaNodes = payload?.result?.num_nodes ?? result?.num_nodes ?? "-";
  const metaEdges = payload?.result?.num_edges ?? result?.num_edges ?? "-";
  const metaTime = executionTime != null ? Number(executionTime).toFixed(4) : null;
  const jobShort = jobId ? String(jobId).slice(0, 8) : null;
  const rawResult = payload?.result ?? result ?? null;

  const effectiveTopK = topK ?? 10;

  const slicedTableData = useMemo(() => {
    const rows = payload?.table_data || [];
    if (!Array.isArray(rows)) return [];
    if (!TOP_K_SUPPORTED.has(algorithmName)) return rows;
    return rows.slice(0, effectiveTopK);
  }, [payload?.table_data, algorithmName, effectiveTopK]);

  const slicedChartData = useMemo(() => {
    const chart = payload?.chart_data || null;
    if (!chart || !TOP_K_SUPPORTED.has(algorithmName)) return chart;

    const labels = Array.isArray(chart.labels) ? chart.labels.slice(0, effectiveTopK) : [];
    const values = Array.isArray(chart.values) ? chart.values.slice(0, effectiveTopK) : [];
    const next = { ...chart, labels, values };
    if (Array.isArray(chart.series_auth)) {
      next.series_auth = chart.series_auth.slice(0, effectiveTopK);
    }
    return next;
  }, [payload?.chart_data, algorithmName, effectiveTopK]);

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Results</h2>
        <p>
          Review tables, charts, and graph highlights for{" "}
          <strong>{algorithmName}</strong>.
        </p>
        <div className="results-meta-row" role="list" aria-label="Run metadata">
          <span className="badge badge--compact" role="listitem">
            Nodes: <strong>{metaNodes}</strong>
          </span>
          <span className="badge badge--compact" role="listitem">
            Edges: <strong>{metaEdges}</strong>
          </span>
          {TOP_K_SUPPORTED.has(algorithmName) ? (
            <span className="badge badge--compact" role="listitem">
              Top-k:{" "}
              <select
                aria-label="Top-k"
                className="badge-select"
                onChange={(e) => setTopK(Number.parseInt(e.target.value, 10))}
                value={effectiveTopK}
              >
                {TOP_K_OPTIONS.map((k) => (
                  <option key={`topk-${k}`} value={k}>{k}</option>
                ))}
              </select>
            </span>
          ) : null}
          {metaTime ? (
            <span className="badge badge--compact badge--ok" role="listitem">
              Time: <strong>{metaTime}s</strong>
            </span>
          ) : null}
          {jobShort ? (
            <span className="badge badge--compact" role="listitem">
              Job: <strong>{jobShort}</strong>
            </span>
          ) : null}
        </div>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}
      {isLoading ? <div className="report-box">Loading results…</div> : null}

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
            tableData={slicedTableData}
            rawResult={rawResult}
            jobParams={payload?.params || null}
          />
        ) : null}
        {activeTab === "chart" ? (
          <ChartView
            algorithmName={algorithmName}
            chartData={slicedChartData}
            rawResult={rawResult}
          />
        ) : null}
        {activeTab === "graph" ? (
          <GraphHighlight
            algorithmName={algorithmName}
            graphData={payload?.graph_viz || null}
          />
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

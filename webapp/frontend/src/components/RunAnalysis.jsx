import { useEffect, useRef, useState } from "react";
import {
  runAlgorithm,
  streamAlgorithmProgress,
} from "../services/api";

/**
 * Step 5 — Execute the algorithm and stream live progress.
 *
 * Pipeline on mount:
 *   1. POST /algorithms/run    → receives job_id
 *   2. GET  /algorithms/run/stream/{job_id} → live NDJSON progress
 *   3. Render progress bar + elapsed-time counter while running
 *   4. On completion, show summary panel with "View Results →" button
 */

function formatAlgorithmSummary(result) {
  if (!result || !result.result) return "";
  const inner = result.result;
  const algo  = result.algorithm;

  if (algo === "louvain") {
    const n = inner.num_communities ?? (inner.top_communities?.length || 0);
    return `Found ${n} communities`;
  }
  if (algo === "mcl") {
    const n = inner.num_clusters ?? new Set(inner.cluster_assignments || []).size;
    return `Found ${n} clusters`;
  }
  if (algo === "pagerank") {
    const top = inner.top_regulators?.[0];
    return top
      ? `Top regulator: ${top.label || `node_${top.index}`}`
      : "PageRank converged";
  }
  if (algo === "rwr") {
    const top = inner.top_nodes?.[0];
    return top
      ? `Closest node: ${top.label || `node_${top.index}`}`
      : "RWR converged";
  }
  if (algo === "hits") {
    const hub = inner.top_hubs?.[0];
    return hub
      ? `Top hub: ${hub.label || `node_${hub.index}`}`
      : "HITS converged";
  }
  if (algo === "bfs") {
    const reached = inner.num_reached ?? (inner.visited_order?.length || 0);
    return `Cascade reached ${reached} nodes`;
  }
  return "Analysis complete";
}

function RunAnalysis({ uploadId, algorithmConfig, onBack, onComplete }) {
  const [phase, setPhase]         = useState("submitting"); // submitting | running | done | error
  const [progress, setProgress]   = useState({ stage: "starting", percent: 0, message: "" });
  const [elapsed, setElapsed]     = useState(0);
  const [jobId, setJobId]         = useState(null);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState("");
  const startedAtRef              = useRef(null);
  const intervalRef               = useRef(null);

  // Tick elapsed-time counter every second while running
  useEffect(() => {
    if (phase !== "submitting" && phase !== "running") {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return undefined;
    }
    if (!startedAtRef.current) {
      startedAtRef.current = Date.now();
    }
    intervalRef.current = setInterval(() => {
      setElapsed((Date.now() - startedAtRef.current) / 1000);
    }, 1000);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [phase]);

  // Kick off the job exactly once on mount
  useEffect(() => {
    let cancelled = false;
    async function go() {
      if (!uploadId || !algorithmConfig) {
        setError("Missing upload or algorithm configuration.");
        setPhase("error");
        return;
      }
      try {
        const submitResp = await runAlgorithm(
          uploadId,
          algorithmConfig.algorithm,
          algorithmConfig.mode,
          algorithmConfig.params,
        );
        if (cancelled) return;
        setJobId(submitResp.job_id);
        setPhase("running");

        await streamAlgorithmProgress(
          submitResp.job_id,
          (event) => {
            if (cancelled) return;
            setProgress({
              stage:   event.stage   || "running",
              percent: event.percent || 0,
              message: event.message || "",
            });
          },
          (resultData) => {
            if (cancelled) return;
            setResult(resultData);
            setProgress((prev) => ({ ...prev, percent: 100, stage: "done" }));
            setPhase("done");
          },
          (err) => {
            if (cancelled) return;
            setError(err.message);
            setPhase("error");
          },
        );
      } catch (err) {
        if (cancelled) return;
        setError(err.message);
        setPhase("error");
      }
    }
    go();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isGpu       = algorithmConfig?.mode === "gpu";
  const percent     = Math.max(0, Math.min(100, Math.round(progress.percent)));
  const stageLabel  = (progress.stage || "running").replaceAll("_", " ");

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Running Analysis</h2>
        <p>
          Executing <strong>{algorithmConfig?.algorithm}</strong> in{" "}
          <strong>{algorithmConfig?.mode}</strong> mode. Live progress streamed
          from the backend.
        </p>
      </div>

      <div className="run-header-row">
        <span className="algo-badge">{algorithmConfig?.algorithm?.toUpperCase()}</span>
        <span className={`mode-badge mode-${algorithmConfig?.mode}`}>
          {algorithmConfig?.mode}
        </span>
        <span
          className={`gpu-active-badge${isGpu && phase === "running" ? " is-active" : " is-cpu"}`}
        >
          {isGpu ? "GPU Active" : "CPU Mode"}
        </span>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      {(phase === "submitting" || phase === "running") && (
        <div className="report-box progress-box">
          <div className="progress-header">
            <h3>Algorithm Progress</h3>
            <strong>{percent}%</strong>
          </div>
          <p className="progress-stage-label">
            Stage: <span>{stageLabel}</span>
            {progress.message ? ` — ${progress.message}` : ""}
          </p>
          <div aria-hidden="true" className="progress-track">
            <div className="progress-fill" style={{ width: `${percent}%` }} />
          </div>
          <p className="progress-copy">
            Elapsed: <span>{elapsed.toFixed(1)}s</span>
          </p>
        </div>
      )}

      {phase === "done" && result ? (
        <div className="report-box summary-box">
          <h3>Analysis Complete</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <span>Execution time</span>
              <strong>{Number(result.execution_time || 0).toFixed(4)}s</strong>
            </div>
            <div className="stat-card">
              <span>Nodes processed</span>
              <strong>{result.num_nodes ?? "-"}</strong>
            </div>
            <div className="stat-card">
              <span>Edges processed</span>
              <strong>{result.num_edges ?? "-"}</strong>
            </div>
          </div>
          <p className="summary-line">{formatAlgorithmSummary(result)}</p>
        </div>
      ) : null}

      <div className="action-row">
        <button
          className="secondary-button"
          disabled={phase === "running" || phase === "submitting"}
          onClick={onBack}
          type="button"
        >
          Back
        </button>
        <button
          className="primary-button"
          disabled={phase !== "done"}
          onClick={() => onComplete?.(jobId, result)}
          type="button"
        >
          View Results →
        </button>
      </div>
    </div>
  );
}

export default RunAnalysis;

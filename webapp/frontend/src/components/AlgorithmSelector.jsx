import { useEffect, useMemo, useState } from "react";
import { getAlgorithmCatalog } from "../services/api";

/**
 * Step 4 — Algorithm selection + parameter configuration.
 *
 * Left panel: cards for each registered algorithm.
 * Right panel: dynamic param form built from the algorithm's param_schema
 *              returned by GET /algorithms/catalog.
 *
 * Props
 * -----
 * uploadId : string
 *   Forwarded to the next step for the actual run.
 * onNext({algorithm, mode, params}) : function
 *   Called when the user clicks "Run Analysis".
 */

// Per-parameter biological tooltips — keyed by `${algoName}.${paramName}`.
const PARAM_TOOLTIPS = {
  "pagerank.damping":
    "Probability of following a regulatory edge vs. teleporting. 0.85 is standard.",
  "pagerank.max_iter":
    "Maximum power iterations before stopping.",
  "bfs.source":
    "Starting node index for cascade tracing. Use a known TF index.",
  "bfs.max_depth":
    "Maximum regulatory cascade depth to explore.",
  "louvain.resolution":
    "Higher values find more, smaller communities.",
  "rwr.restart_prob":
    "Probability of returning to seed TF at each step.",
  "rwr.seed_nodes":
    "Comma-separated node indices for seed TFs.",
  "hits.max_iter":
    "Maximum HITS iterations.",
  "mcl.inflation":
    "Controls cluster granularity. Higher = more clusters.",
};

// Friendly one-line descriptions for the algorithm cards.
const ALGO_DESCRIPTIONS = {
  pagerank: "Rank regulators by global influence in the network.",
  bfs:      "Trace a regulatory cascade outward from a source TF.",
  louvain:  "Detect communities of co-regulated genes.",
  rwr:      "Find nodes proximal to a set of seed TFs.",
  hits:     "Identify hubs and authorities (transcription factors vs targets).",
  mcl:      "Markov-clustering for tightly co-regulated modules.",
};

// Hardcoded display order — matches the order in the spec.
const ALGO_DISPLAY_ORDER = ["pagerank", "bfs", "louvain", "rwr", "hits", "mcl"];

function defaultValueFor(paramDef) {
  if (paramDef.default !== undefined && paramDef.default !== null) {
    return paramDef.default;
  }
  if (paramDef.type === "int" || paramDef.type === "integer") return 0;
  if (paramDef.type === "float" || paramDef.type === "number") return 0.0;
  return "";
}

function paramFormDefaults(schema) {
  const defaults = {};
  for (const [key, def] of Object.entries(schema || {})) {
    defaults[key] = defaultValueFor(def);
  }
  return defaults;
}

function AlgorithmSelector({ uploadId, onBack, onNext }) {
  const [catalog, setCatalog]   = useState([]);
  const [isLoading, setLoading] = useState(false);
  const [error, setError]       = useState("");
  const [selected, setSelected] = useState(null);
  const [mode, setMode]         = useState("gpu");
  const [params, setParams]     = useState({});

  // Load catalog once on mount
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    getAlgorithmCatalog()
      .then((data) => {
        if (!active) return;
        setCatalog(data || []);
        // Pre-select pagerank if available
        const initial =
          data.find((a) => a.name === "pagerank") || data[0] || null;
        if (initial) {
          setSelected(initial.name);
          setParams(paramFormDefaults(initial.param_schema));
        }
      })
      .catch((err) => {
        if (active) setError(err.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  // Sorted algorithm list using the canonical display order
  const orderedCatalog = useMemo(() => {
    const byName = Object.fromEntries(catalog.map((a) => [a.name, a]));
    return ALGO_DISPLAY_ORDER
      .map((name) => byName[name])
      .filter(Boolean);
  }, [catalog]);

  const selectedAlgo = useMemo(
    () => catalog.find((a) => a.name === selected) || null,
    [catalog, selected],
  );

  function handleSelectAlgorithm(algo) {
    setSelected(algo.name);
    setParams(paramFormDefaults(algo.param_schema));
  }

  function handleParamChange(key, rawValue, paramDef) {
    let value = rawValue;
    if (paramDef.type === "int" || paramDef.type === "integer") {
      value = rawValue === "" ? "" : Number.parseInt(rawValue, 10);
    } else if (paramDef.type === "float" || paramDef.type === "number") {
      value = rawValue === "" ? "" : Number.parseFloat(rawValue);
    }
    setParams((prev) => ({ ...prev, [key]: value }));
  }

  function handleRun() {
    if (!selectedAlgo) return;
    onNext?.({
      algorithm: selectedAlgo.name,
      mode,
      params: { ...params },
    });
  }

  return (
    <div className="panel-section">
      <div className="section-header">
        <h2>Choose Algorithm</h2>
        <p>
          Select one of the six GPU-accelerated algorithms and tune its
          parameters before launching the run.
        </p>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}
      {isLoading ? (
        <div className="report-box">Loading algorithm catalog…</div>
      ) : null}

      <div className="algorithm-layout">
        <div className="algorithm-cards-grid">
          {orderedCatalog.map((algo) => (
            <button
              key={algo.name}
              className={`algorithm-card${
                selected === algo.name ? " is-selected" : ""
              }`}
              onClick={() => handleSelectAlgorithm(algo)}
              type="button"
            >
              <strong>{algo.name.toUpperCase()}</strong>
              <span>
                {ALGO_DESCRIPTIONS[algo.name] || algo.description || ""}
              </span>
            </button>
          ))}
        </div>

        <div className="param-form">
          <h3>{selectedAlgo ? `${selectedAlgo.name} parameters` : "Parameters"}</h3>

          {!selectedAlgo ? (
            <p className="param-form-empty">Select an algorithm to view its parameters.</p>
          ) : (
            <>
              {Object.entries(selectedAlgo.param_schema || {}).map(
                ([key, def]) => {
                  const tooltipKey = `${selectedAlgo.name}.${key}`;
                  const tooltip = PARAM_TOOLTIPS[tooltipKey] || def.description || "";
                  const inputType =
                    def.type === "int" || def.type === "integer"
                      ? "number"
                      : def.type === "float" || def.type === "number"
                        ? "number"
                        : "text";
                  const step =
                    def.type === "float" || def.type === "number"
                      ? "any"
                      : undefined;
                  return (
                    <label key={key} className="param-field">
                      <span className="param-field-label">
                        {key}
                        {tooltip ? (
                          <em className="param-field-tooltip" title={tooltip}>
                            ⓘ
                          </em>
                        ) : null}
                      </span>
                      <input
                        max={def.max ?? undefined}
                        min={def.min ?? undefined}
                        onChange={(event) =>
                          handleParamChange(key, event.target.value, def)
                        }
                        step={step}
                        type={inputType}
                        value={params[key] ?? ""}
                      />
                      {tooltip ? (
                        <small className="param-field-hint">{tooltip}</small>
                      ) : null}
                    </label>
                  );
                },
              )}

              <label className="param-field">
                <span className="param-field-label">Execution mode</span>
                <select
                  onChange={(event) => setMode(event.target.value)}
                  value={mode}
                >
                  <option value="gpu">gpu (recommended)</option>
                  <option value="cpu_single">cpu_single</option>
                  <option value="cpu_multi">cpu_multi</option>
                </select>
                <small className="param-field-hint">
                  GPU mode falls back to cpu_single if no CUDA device is present.
                </small>
              </label>
            </>
          )}
        </div>
      </div>

      <div className="action-row">
        <button className="secondary-button" onClick={onBack} type="button">
          Back
        </button>
        <button
          className="primary-button"
          disabled={!selectedAlgo || !uploadId}
          onClick={handleRun}
          type="button"
        >
          Run Analysis
        </button>
      </div>
    </div>
  );
}

export default AlgorithmSelector;

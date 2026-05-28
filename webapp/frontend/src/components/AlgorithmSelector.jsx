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

// Small chips to reinforce “biological network analysis” framing.
const ALGO_CHIPS = {
  pagerank: ["Centrality", "Ranking"],
  bfs:      ["Cascade", "Reachability"],
  louvain:  ["Communities", "Modularity"],
  rwr:      ["Diffusion", "Seeds"],
  hits:     ["Hubs", "Authorities"],
  mcl:      ["Clustering", "Markov"],
};

// Hardcoded display order — matches the order in the spec.
const ALGO_DISPLAY_ORDER = ["pagerank", "bfs", "louvain", "rwr", "hits", "mcl"];

function defaultValueFor(paramDef) {
  // Backend currently returns a plain dict of default values (not typed defs).
  // Support both shapes: primitive defaults or rich {default,type,...} objects.
  if (paramDef === null || paramDef === undefined) {
    return "";
  }
  if (Array.isArray(paramDef)) {
    // Represent list params as comma-separated text in the UI.
    return "";
  }
  if (typeof paramDef !== "object") {
    return paramDef;
  }
  if (Array.isArray(paramDef.default)) {
    return "";
  }
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

function inferParamType(def) {
  if (Array.isArray(def)) {
    return "list";
  }
  if (typeof def === "object" && def && def.type) {
    return def.type;
  }
  if (typeof def === "number") {
    return Number.isInteger(def) ? "int" : "float";
  }
  return "text";
}

function parseIndexList(text) {
  if (text === null || text === undefined) return [];
  const raw = String(text).trim();
  if (!raw) return [];
  return raw
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Number.parseInt(s, 10))
    .filter((n) => Number.isFinite(n));
}

function AlgorithmSelector({ uploadId, networkType, onBack, onNext }) {
  const [catalog, setCatalog]   = useState([]);
  const [isLoading, setLoading] = useState(false);
  const [error, setError]       = useState("");
  const [selected, setSelected] = useState(null);
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
          const defaults = paramFormDefaults(initial.param_schema);
          setParams({
            ...defaults,
            network_type: networkType || defaults.network_type,
          });
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
    const defaults = paramFormDefaults(algo.param_schema);
    setParams({
      ...defaults,
      network_type: networkType || defaults.network_type,
    });
  }

  function handleParamChange(key, rawValue, paramDef) {
    let value = rawValue;
    const inferredType = inferParamType(paramDef);
    if (inferredType === "int" || inferredType === "integer") {
      value = rawValue === "" ? "" : Number.parseInt(rawValue, 10);
    } else if (inferredType === "float" || inferredType === "number") {
      value = rawValue === "" ? "" : Number.parseFloat(rawValue);
    } else if (inferredType === "list") {
      // Keep as text in form state; parse on submit.
      value = rawValue;
    }
    setParams((prev) => ({ ...prev, [key]: value }));
  }

  function handleRun() {
    if (!selectedAlgo) return;
    const schema = selectedAlgo.param_schema || {};
    const normalized = {};
    for (const [key, def] of Object.entries(schema)) {
      if (key === "network_type") continue;
      const inferred = inferParamType(def);
      const v = params[key];
      if (inferred === "list") {
        normalized[key] = parseIndexList(v);
      } else {
        normalized[key] = v;
      }
    }

    onNext?.({
      algorithm: selectedAlgo.name,
      mode: "gpu",
      params: {
        ...normalized,
        network_type: networkType || params.network_type || "grn",
      },
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
              <div className="algorithm-card-header">
                <strong>{algo.name.toUpperCase()}</strong>
                <span className="algorithm-card-badge">GPU</span>
              </div>
              <div className="algorithm-chip-row" aria-label="Algorithm category">
                {(ALGO_CHIPS[algo.name] || []).map((chip) => (
                  <span className="chip" key={`${algo.name}-${chip}`}>{chip}</span>
                ))}
              </div>
              <span className="algorithm-card-desc">
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
                  if (key === "network_type") {
                    return null;
                  }
                  const tooltipKey = `${selectedAlgo.name}.${key}`;
                  const tooltip =
                    PARAM_TOOLTIPS[tooltipKey] ||
                    (typeof def === "object" && def ? def.description : "") ||
                    "";

                  const inferredType = inferParamType(def);

                  const placeholder =
                    inferredType === "list"
                      ? "e.g. 12, 48, 102"
                      : undefined;

                  const inputType =
                    inferredType === "int" || inferredType === "integer" || inferredType === "float" || inferredType === "number"
                      ? "number"
                      : "text";
                  const step =
                    inferredType === "float" || inferredType === "number"
                      ? "any"
                      : undefined;
                  const min = typeof def === "object" && def ? def.min : undefined;
                  const max = typeof def === "object" && def ? def.max : undefined;
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
                        max={max ?? undefined}
                        min={min ?? undefined}
                        onChange={(event) =>
                          handleParamChange(key, event.target.value, def)
                        }
                        placeholder={placeholder}
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

              <div className="param-field param-field--readonly">
                <span className="param-field-label">Network type</span>
                <span className="param-field-value-fixed">
                  {String(networkType || "grn").toUpperCase()}
                </span>
                <small className="param-field-hint">
                  Set on the Upload step.
                </small>
              </div>

              <div className="param-field param-field--readonly">
                <span className="param-field-label">Execution mode</span>
                <span className="param-field-value-fixed">GPU (CUDA)</span>
                <small className="param-field-hint">
                  Analyses always run on the detected NVIDIA GPU.
                </small>
              </div>
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

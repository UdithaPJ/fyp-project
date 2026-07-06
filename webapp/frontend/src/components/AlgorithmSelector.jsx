import { useEffect, useMemo, useRef, useState } from "react";
import { getAlgorithmCatalog, getNodes } from "../services/api";

/**
 * Step 4 — Algorithm selection + parameter configuration.
 *
 * The form is built dynamically from the backend's `ui_schema` field
 * (added to GET /algorithms/catalog) — one renderer per param `type`:
 *
 *   slider              → ParamSlider              (range + live value)
 *   number              → ParamNumber              (text input, validated)
 *   select              → ParamSelect              (<select> dropdown)
 *   preset_select       → ParamPresetSelect        (pill row → float)
 *   node_selector       → ParamNodeSelector        (typeahead → index)
 *   multi_node_selector → ParamMultiNodeSelector   (typeahead → indices)
 *
 * Standard params render unconditionally.  Params with `advanced: true`
 * are hidden behind a collapsible "⚙ Advanced Settings" toggle.
 *
 * paramValues always holds raw float / int / list values — preset
 * pills set the float directly when clicked, never the label.
 *
 * Props (unchanged from before):
 *   uploadId : string
 *   onBack   : function
 *   onNext({ algorithm, mode, params }) : function
 */

// Per-parameter biological tooltips — keyed by `${algoName}.${paramName}`.
const PARAM_TOOLTIPS = {
  "pagerank.damping":
    "Probability of following a regulatory edge vs. teleporting. 0.85 is standard.",
  "pagerank.max_iter": "Maximum power iterations before stopping.",
  "bfs.source":
    "Starting node index for cascade tracing. Use a known TF index.",
  "bfs.max_depth": "Maximum regulatory cascade depth to explore.",
  "louvain.resolution": "Higher values find more, smaller communities.",
  "rwr.restart_prob": "Probability of returning to seed TF at each step.",
  "rwr.seed_nodes": "Comma-separated node indices for seed TFs.",
  "hits.max_iter": "Maximum HITS iterations.",
  "mcl.inflation": "Controls cluster granularity. Higher = more clusters.",
};

// Friendly one-line descriptions for the algorithm cards.
const ALGO_DESCRIPTIONS = {
  pagerank: "Rank regulators by global influence in the network.",
  bfs: "Trace a regulatory cascade outward from a source TF.",
  louvain: "Detect communities of co-regulated genes.",
  rwr: "Find nodes proximal to a set of seed TFs.",
  hits: "Identify hubs and authorities (transcription factors vs targets).",
  mcl: "Markov-clustering for tightly co-regulated modules.",
};

// Small chips to reinforce “biological network analysis” framing.
const ALGO_CHIPS = {
  pagerank: ["Centrality", "Ranking"],
  bfs: ["Cascade", "Reachability"],
  louvain: ["Communities", "Modularity"],
  rwr: ["Diffusion", "Seeds"],
  hits: ["Hubs", "Authorities"],
  mcl: ["Clustering", "Markov"],
};

// Hardcoded display order — matches the order in the spec.
const ALGO_DISPLAY_ORDER = ["pagerank", "bfs", "louvain", "rwr", "hits", "mcl"];
const DEFAULT_TYPEAHEAD_LIMIT = 50;
const TYPEAHEAD_DEBOUNCE_MS = 250;

// ===========================================================================
// Helpers
// ===========================================================================

function buildDefaultValues(uiSchema) {
  // Resolve a default value for every param in the schema. Used on
  // algorithm selection AND on reset.
  const out = {};
  for (const p of uiSchema || []) {
    out[p.key] = p.default;
  }
  return out;
}

function formatValueForDisplay(param, value) {
  if (value === null || value === undefined || value === "") return "";
  if (param.display_format === "scientific" && typeof value === "number") {
    return value.toExponential(1);
  }
  if (typeof value === "number") {
    // Show ints as ints; floats with up to 4 significant digits
    if (Number.isInteger(value)) return String(value);
    if (Math.abs(value) < 1e-3 || Math.abs(value) >= 10000) {
      return value.toExponential(2);
    }
    return value.toFixed(2).replace(/\.?0+$/, "");
  }
  return String(value);
}

function parseNumericInput(value, paramType) {
  if (value === "" || value === null || value === undefined) return "";
  const asFloat = Number.parseFloat(value);
  if (Number.isNaN(asFloat)) return value;
  return asFloat;
}

// ===========================================================================
// Tooltip — small ⓘ icon with hover-popup
// ===========================================================================

function ParamTooltip({ text }) {
  const [visible, setVisible] = useState(false);
  if (!text) return null;
  return (
    <span className="param-tooltip-wrapper">
      <span
        className="param-tooltip-icon"
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        aria-label="Parameter help"
      >
        ?
      </span>
      {visible ? <div className="param-tooltip-popup">{text}</div> : null}
    </span>
  );
}

// ===========================================================================
// Slider
// ===========================================================================

function ParamSlider({ param, value, onChange }) {
  const safeValue = typeof value === "number" ? value : param.default;
  return (
    <div className="param-field">
      <div className="param-header">
        <label htmlFor={`p-${param.key}`} className="param-field-label">
          {param.label}
        </label>
        <ParamTooltip text={param.tooltip} />
        <span className="param-value-display">
          {formatValueForDisplay(param, safeValue)}
        </span>
      </div>
      <input
        id={`p-${param.key}`}
        type="range"
        min={param.min}
        max={param.max}
        step={param.step ?? "any"}
        value={safeValue}
        onChange={(e) => onChange(Number.parseFloat(e.target.value))}
        className="param-slider"
      />
      <div className="param-range-labels">
        <span>{formatValueForDisplay(param, param.min)}</span>
        <span>{formatValueForDisplay(param, param.max)}</span>
      </div>
    </div>
  );
}

// ===========================================================================
// Number
// ===========================================================================

function ParamNumber({ param, value, onChange }) {
  return (
    <div className="param-field">
      <div className="param-header">
        <label htmlFor={`p-${param.key}`} className="param-field-label">
          {param.label}
        </label>
        <ParamTooltip text={param.tooltip} />
      </div>
      <input
        id={`p-${param.key}`}
        type="number"
        min={param.min}
        max={param.max}
        step={param.step ?? "any"}
        value={value ?? ""}
        onChange={(e) => onChange(parseNumericInput(e.target.value, "number"))}
        className="param-number-input"
      />
    </div>
  );
}

// ===========================================================================
// Select (dropdown with known options)
// ===========================================================================

function ParamSelect({ param, value, onChange }) {
  const numericOptions = (param.options || []).every(
    (o) => typeof o.value === "number",
  );
  return (
    <div className="param-field">
      <div className="param-header">
        <label htmlFor={`p-${param.key}`} className="param-field-label">
          {param.label}
        </label>
        <ParamTooltip text={param.tooltip} />
      </div>
      <select
        id={`p-${param.key}`}
        className="param-select"
        value={value ?? param.default}
        onChange={(e) => {
          const raw = e.target.value;
          onChange(numericOptions ? Number(raw) : raw);
        }}
      >
        {(param.options || []).map((opt) => (
          <option key={String(opt.value)} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}

// ===========================================================================
// Preset pills — pills are display affordances; underlying value is a float
// ===========================================================================

function ParamPresetSelect({ param, value, onChange }) {
  const activePreset = (param.presets || []).find((p) => p.value === value);
  return (
    <div className="param-field">
      <div className="param-header">
        <label className="param-field-label">{param.label}</label>
        <ParamTooltip text={param.tooltip} />
      </div>
      <div className="preset-pills">
        {(param.presets || []).map((preset) => {
          const isActive = activePreset?.label === preset.label;
          return (
            <button
              key={preset.label}
              type="button"
              className={`preset-pill${isActive ? " active" : ""}`}
              onClick={() => onChange(preset.value)}
            >
              {preset.label}
            </button>
          );
        })}
      </div>
      <span className="preset-raw-value">
        Value: {typeof value === "number" ? value.toExponential(0) : "—"}
      </span>
    </div>
  );
}

// ===========================================================================
// Node selector (single) — typeahead, returns the picked node's index
// ===========================================================================

function useDebouncedQuery(query, delay = TYPEAHEAD_DEBOUNCE_MS) {
  const [debounced, setDebounced] = useState(query);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(query), delay);
    return () => clearTimeout(id);
  }, [query, delay]);
  return debounced;
}

function useNodeSearch(uploadId, query, initialOptions) {
  const [results, setResults] = useState(
    Array.isArray(initialOptions) ? initialOptions : [],
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const debouncedQuery = useDebouncedQuery(query);
  const cachedOptions = useMemo(
    () => (Array.isArray(initialOptions) ? initialOptions : []),
    [initialOptions],
  );

  useEffect(() => {
    if (!debouncedQuery) {
      setResults(cachedOptions);
    }
  }, [cachedOptions, debouncedQuery]);

  useEffect(() => {
    let cancelled = false;
    if (!uploadId) {
      setResults([]);
      setError("");
      return undefined;
    }
    setLoading(true);
    setError("");
    getNodes(uploadId, debouncedQuery, DEFAULT_TYPEAHEAD_LIMIT)
      .then((res) => {
        if (cancelled) return;
        setResults(Array.isArray(res?.nodes) ? res.nodes : []);
      })
      .catch((err) => {
        if (cancelled) return;
        setResults(!debouncedQuery ? cachedOptions : []);
        setError(err?.message || "Unable to load node names.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [uploadId, debouncedQuery, cachedOptions]);

  return { results, loading, error };
}

function ParamNodeSelector({ param, value, onChange, uploadId, initialNodes }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const { results, loading, error } = useNodeSearch(uploadId, query, initialNodes);

  // Look up the label for the currently selected index.
  const selectedNode =
    initialNodes.find((n) => n.index === value) ||
    results.find((n) => n.index === value);
  const selectedLabel = selectedNode
    ? selectedNode.label
    : value === 0 || typeof value === "number"
      ? `Node ${value}`
      : "Select a node…";

  const containerRef = useRef(null);
  useEffect(() => {
    function onDocClick(e) {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  return (
    <div className="param-field">
      <div className="param-header">
        <label className="param-field-label">{param.label}</label>
        <ParamTooltip text={param.tooltip} />
      </div>
      <div className="node-selector" ref={containerRef}>
        <div
          className="node-selector-display"
          onClick={() => setOpen((o) => !o)}
          role="button"
          tabIndex={0}
        >
          <span className="selected-node-label">{selectedLabel}</span>
          <span className="node-selector-chevron">▼</span>
        </div>
        {open ? (
          <div className="node-dropdown">
            <input
              type="text"
              placeholder={param.placeholder || "Search nodes…"}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="node-search-input"
              autoFocus
            />
            {error ? <div className="node-error">{error}</div> : null}
            {loading ? <div className="node-loading">Searching…</div> : null}
            <div className="node-list">
              {results.length === 0 && !loading && !error ? (
                <div className="node-loading">No matches.</div>
              ) : null}
              {results.map((node) => (
                <div
                  key={node.index}
                  className={`node-option${
                    node.index === value ? " selected" : ""
                  }`}
                  onClick={() => {
                    onChange(node.index);
                    setOpen(false);
                    setQuery("");
                  }}
                >
                  <span className="node-label">{node.label}</span>
                  <span className="node-index">#{node.index}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ===========================================================================
// Multi-node selector — pills above the search box
// ===========================================================================

function ParamMultiNodeSelector({
  param,
  value,
  onChange,
  uploadId,
  initialNodes,
}) {
  const selectedIndices = Array.isArray(value) ? value : [];
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const { results, loading, error } = useNodeSearch(uploadId, query, initialNodes);

  // Resolve labels for the selected indices: prefer initialNodes,
  // then fall back to the search results, then a placeholder.
  const knownByIndex = useMemo(() => {
    const map = new Map();
    initialNodes.forEach((n) => map.set(n.index, n.label));
    results.forEach((n) => map.set(n.index, n.label));
    return map;
  }, [initialNodes, results]);

  function addIndex(idx) {
    if (selectedIndices.includes(idx)) return;
    onChange([...selectedIndices, idx]);
  }
  function removeIndex(idx) {
    onChange(selectedIndices.filter((i) => i !== idx));
  }

  const containerRef = useRef(null);
  useEffect(() => {
    function onDocClick(e) {
      if (!containerRef.current) return;
      if (!containerRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  return (
    <div className="param-field">
      <div className="param-header">
        <label className="param-field-label">{param.label}</label>
        <ParamTooltip text={param.tooltip} />
      </div>
      <div className="multi-node-selector" ref={containerRef}>
        <div className="selected-nodes-pills">
          {selectedIndices.length === 0 ? (
            <span className="no-seeds-hint">
              No seeds selected — will use all nodes equally
            </span>
          ) : (
            selectedIndices.map((idx) => (
              <span key={idx} className="selected-node-pill">
                {knownByIndex.get(idx) || `Node ${idx}`}
                <button
                  type="button"
                  className="pill-remove"
                  onClick={() => removeIndex(idx)}
                  aria-label={`Remove node ${idx}`}
                >
                  ×
                </button>
              </span>
            ))
          )}
        </div>
        <div
          className="node-selector-display"
          onClick={() => setOpen((o) => !o)}
          role="button"
          tabIndex={0}
        >
          <span className="selected-node-label">
            {open ? "Selecting…" : "Add seed nodes…"}
          </span>
          <span className="node-selector-chevron">▼</span>
        </div>
        {open ? (
          <div className="node-dropdown">
            <input
              type="text"
              placeholder={param.placeholder || "Search nodes…"}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="node-search-input"
              autoFocus
            />
            {error ? <div className="node-error">{error}</div> : null}
            {loading ? <div className="node-loading">Searching…</div> : null}
            <div className="node-list">
              {results.length === 0 && !loading && !error ? (
                <div className="node-loading">No matches.</div>
              ) : null}
              {results.map((node) => (
                <div
                  key={node.index}
                  className={`node-option${
                    selectedIndices.includes(node.index) ? " selected" : ""
                  }`}
                  onClick={() => {
                    addIndex(node.index);
                    setQuery("");
                  }}
                >
                  <span className="node-label">{node.label}</span>
                  <span className="node-index">#{node.index}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

// ===========================================================================
// Dispatcher — pick the renderer for a param.type
// ===========================================================================

function ParamRenderer({ param, value, onChange, uploadId, initialNodes }) {
  switch (param.type) {
    case "slider":
      return <ParamSlider param={param} value={value} onChange={onChange} />;
    case "number":
      return <ParamNumber param={param} value={value} onChange={onChange} />;
    case "select":
      return <ParamSelect param={param} value={value} onChange={onChange} />;
    case "preset_select":
      return (
        <ParamPresetSelect param={param} value={value} onChange={onChange} />
      );
    case "node_selector":
      return (
        <ParamNodeSelector
          param={param}
          value={value}
          onChange={onChange}
          uploadId={uploadId}
          initialNodes={initialNodes}
        />
      );
    case "multi_node_selector":
      return (
        <ParamMultiNodeSelector
          param={param}
          value={value}
          onChange={onChange}
          uploadId={uploadId}
          initialNodes={initialNodes}
        />
      );
    default:
      // Unknown type — fall back to a plain numeric input.
      return <ParamNumber param={param} value={value} onChange={onChange} />;
  }
}

// ===========================================================================
// Top-level component
// ===========================================================================

function AlgorithmSelector({ uploadId, onBack, onNext, networkType }) {
  const [catalog, setCatalog] = useState([]);
  const [isLoading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedName, setSelectedName] = useState(null);
  const [paramValues, setParamValues] = useState({});
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [initialNodes, setInitialNodes] = useState([]);

  // Load catalog once on mount
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    getAlgorithmCatalog()
      .then((data) => {
        if (!active) return;
        const list = Array.isArray(data) ? data : [];
        setCatalog(list);
        const initial =
          list.find((a) => a.name === "pagerank") || list[0] || null;
        if (initial) {
          setSelectedName(initial.name);
          setParamValues(buildDefaultValues(initial.ui_schema));
          setAdvancedOpen(false);
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

  // Prefetch the first 50 nodes for typeahead initial options.
  useEffect(() => {
    if (!uploadId) return;
    let active = true;
    getNodes(uploadId, "", DEFAULT_TYPEAHEAD_LIMIT)
      .then((res) => {
        if (!active) return;
        setInitialNodes(Array.isArray(res?.nodes) ? res.nodes : []);
      })
      .catch(() => {
        if (active) setInitialNodes([]);
      });
    return () => {
      active = false;
    };
  }, [uploadId]);

  const orderedCatalog = useMemo(() => {
    const byName = Object.fromEntries(catalog.map((a) => [a.name, a]));
    return ALGO_DISPLAY_ORDER.map((name) => byName[name]).filter(Boolean);
  }, [catalog]);

  const selectedAlgo = useMemo(
    () => catalog.find((a) => a.name === selectedName) || null,
    [catalog, selectedName],
  );

  function handleSelectAlgorithm(algo) {
    setSelectedName(algo.name);
    setParamValues(buildDefaultValues(algo.ui_schema));
    setAdvancedOpen(false);
  }

  function setParam(key, newValue) {
    setParamValues((prev) => ({ ...prev, [key]: newValue }));
  }

  function handleSubmit() {
    if (!selectedAlgo) return;
    // paramValues is already raw — presets resolved to floats on click,
    // node selectors resolved to indices, sliders resolved to numbers.
    onNext?.({
      algorithm: selectedAlgo.name,
      mode: "gpu",
      params: { ...paramValues },
    });
  }

  // Split ui_schema into standard vs advanced params.
  const uiSchema = selectedAlgo?.ui_schema || [];
  const standardParams = uiSchema.filter((p) => !p.advanced);
  const advancedParams = uiSchema.filter((p) => p.advanced);

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
                selectedName === algo.name ? " is-selected" : ""
              }`}
              onClick={() => handleSelectAlgorithm(algo)}
              type="button"
            >
              {algo.category ? (
                <span className="algo-tag">{algo.category}</span>
              ) : null}
              <strong>{algo.display_name || algo.name.toUpperCase()}</strong>
              <span className="algo-desc">
                {algo.description ||
                  "GPU-accelerated network analysis algorithm."}
              </span>
            </button>
          ))}
        </div>

        <div className="param-form">
          <h3>
            {selectedAlgo
              ? selectedAlgo.display_name || `${selectedAlgo.name} parameters`
              : "Parameters"}
          </h3>
          {selectedAlgo?.description ? (
            <p className="algo-full-desc">{selectedAlgo.description}</p>
          ) : null}

          {!selectedAlgo ? (
            <p className="param-form-empty">
              Select an algorithm to view its parameters.
            </p>
          ) : (
            <>
              {standardParams.map((p) => (
                <ParamRenderer
                  key={p.key}
                  param={p}
                  value={paramValues[p.key]}
                  onChange={(v) => setParam(p.key, v)}
                  uploadId={uploadId}
                  initialNodes={initialNodes}
                />
              ))}

              {advancedParams.length > 0 ? (
                <>
                  <button
                    type="button"
                    className="advanced-toggle"
                    onClick={() => setAdvancedOpen((o) => !o)}
                  >
                    <span>⚙ Advanced Settings</span>
                    <span className="chevron">{advancedOpen ? "▲" : "▼"}</span>
                  </button>
                  {advancedOpen ? (
                    <div className="advanced-params">
                      {advancedParams.map((p) => (
                        <ParamRenderer
                          key={p.key}
                          param={p}
                          value={paramValues[p.key]}
                          onChange={(v) => setParam(p.key, v)}
                          uploadId={uploadId}
                          initialNodes={initialNodes}
                        />
                      ))}
                    </div>
                  ) : null}
                </>
              ) : null}

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
          onClick={handleSubmit}
          type="button"
        >
          Run Analysis →
        </button>
      </div>
    </div>
  );
}

export default AlgorithmSelector;

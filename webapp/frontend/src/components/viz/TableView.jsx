import { useMemo, useState } from "react";

/**
 * Generic sortable table renderer for any algorithm result.
 *
 * Props
 * -----
 * tableData : Array<Object>
 *   Rows returned by the backend's make_*_table helpers.
 * algorithmName : string
 *   Used to special-case BFS cascade rows (group by depth).
 */

function formatHeader(key) {
  return key
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatCell(value) {
  if (value === null || value === undefined) return "-";
  if (Array.isArray(value)) {
    // top_members style — show as comma-separated string
    return value.join(", ");
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(6);
  }
  return String(value);
}

function unsupportedNotice(table) {
  if (!Array.isArray(table) || table.length === 0) return null;
  const first = table[0];
  if (first && first.unsupported) {
    return first.reason || "Table view is not available for this algorithm.";
  }
  return null;
}

// ---------------------------------------------------------------------------
// BFS cascade view — group rows by depth with collapsible sections
// ---------------------------------------------------------------------------

function CascadeTable({ tableData }) {
  const [openDepths, setOpenDepths] = useState(() => new Set([0, 1]));

  function toggle(depth) {
    setOpenDepths((prev) => {
      const next = new Set(prev);
      if (next.has(depth)) next.delete(depth);
      else next.add(depth);
      return next;
    });
  }

  return (
    <div className="cascade-table-wrap">
      {tableData.map((row) => {
        const isOpen = openDepths.has(row.depth);
        return (
          <div key={row.depth} className="cascade-section">
            <button
              className="cascade-section-header"
              onClick={() => toggle(row.depth)}
              type="button"
            >
              <strong>Depth {row.depth}</strong>
              <span>{row.num_nodes} nodes</span>
              <span className="cascade-toggle">{isOpen ? "▾" : "▸"}</span>
            </button>
            {isOpen ? (
              <div className="cascade-section-body">
                {(row.nodes || []).map((n) => (
                  <span key={`${row.depth}-${n}`} className="cascade-chip">
                    {n}
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generic sortable table — used for score / cluster tables
// ---------------------------------------------------------------------------

function GenericSortableTable({ tableData }) {
  const [sortKey, setSortKey]     = useState(null);
  const [sortAsc, setSortAsc]     = useState(true);
  const [showAll, setShowAll]     = useState(false);

  const columns = useMemo(() => {
    // Union of all keys, preserving first-row order
    const cols = [];
    const seen = new Set();
    for (const row of tableData) {
      for (const k of Object.keys(row)) {
        if (!seen.has(k)) {
          cols.push(k);
          seen.add(k);
        }
      }
    }
    return cols;
  }, [tableData]);

  const sortedRows = useMemo(() => {
    if (!sortKey) return tableData;
    const copy = [...tableData];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return copy;
  }, [tableData, sortKey, sortAsc]);

  const displayRows = showAll ? sortedRows.slice(0, 50) : sortedRows.slice(0, 10);

  function handleSort(key) {
    if (sortKey === key) {
      setSortAsc((prev) => !prev);
    } else {
      setSortKey(key);
      setSortAsc(false); // first click = descending (numeric scores)
    }
  }

  return (
    <>
      <div className="preview-table-wrap">
        <table className="preview-table results-table">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c}
                  className="results-table-header"
                  onClick={() => handleSort(c)}
                >
                  {formatHeader(c)}
                  {sortKey === c ? (sortAsc ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {displayRows.map((row, idx) => (
              <tr key={idx} className={idx === 0 ? "results-row-top" : ""}>
                {columns.map((c) => (
                  <td key={c}>
                    {idx === 0 && c === columns[0] ? (
                      <span className="results-top-badge">🏆</span>
                    ) : null}
                    {formatCell(row[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sortedRows.length > 10 ? (
        <div className="action-row">
          <button
            className="secondary-button"
            onClick={() => setShowAll((p) => !p)}
            type="button"
          >
            {showAll ? "Show top 10" : `Show more (${Math.min(50, sortedRows.length)})`}
          </button>
        </div>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------------------
// Main export
// ---------------------------------------------------------------------------

function TableView({ tableData, algorithmName }) {
  const unsupported = unsupportedNotice(tableData);
  if (unsupported) {
    return <div className="viz-empty">{unsupported}</div>;
  }
  if (!Array.isArray(tableData) || tableData.length === 0) {
    return <div className="viz-empty">No table data available.</div>;
  }

  if (algorithmName === "bfs") {
    return <CascadeTable tableData={tableData} />;
  }
  return <GenericSortableTable tableData={tableData} />;
}

export default TableView;

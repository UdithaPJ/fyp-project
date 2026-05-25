import { useMemo } from "react";

/**
 * Horizontal bar chart rendered as inline SVG — no external chart library.
 *
 * Props
 * -----
 * chartData : {chart_type, labels, values, title, x_label, y_label, ...}
 *   Output of src/visualization/charts.make_*_chart_data.
 */

const MAX_BARS  = 20;
const ROW_HEIGHT = 22;
const ROW_GAP    = 6;
const LEFT_LABEL_WIDTH = 140;
const RIGHT_PADDING    = 60;
const CHART_HEIGHT = 400;

function interpolateColor(t) {
  // t in [0,1]: pale teal → deep teal
  const r = Math.round(140 + (47   - 140) * t);
  const g = Math.round(190 + (111  - 190) * t);
  const b = Math.round(205 + (126  - 205) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

function ChartView({ chartData }) {
  const isUnsupported =
    !chartData ||
    chartData.unsupported ||
    chartData.chart_type === "none" ||
    !Array.isArray(chartData.labels) ||
    chartData.labels.length === 0;

  const bars = useMemo(() => {
    if (isUnsupported) return [];
    const pairs = (chartData.labels || []).map((label, i) => ({
      label,
      value: Number(chartData.values?.[i] ?? 0),
    }));
    pairs.sort((a, b) => b.value - a.value);
    return pairs.slice(0, MAX_BARS);
  }, [chartData, isUnsupported]);

  if (isUnsupported) {
    return (
      <div className="viz-empty">
        Chart view is not available for this algorithm.
      </div>
    );
  }

  const maxValue = Math.max(...bars.map((b) => b.value), 1e-9);
  const totalHeight = bars.length * (ROW_HEIGHT + ROW_GAP) + 60;
  const heightStr = `${Math.max(CHART_HEIGHT, totalHeight)}px`;

  return (
    <div className="chart-view-wrap">
      {chartData.title ? <h3 className="chart-title">{chartData.title}</h3> : null}

      <div className="chart-svg-wrap" style={{ height: heightStr }}>
        <svg
          height="100%"
          preserveAspectRatio="none"
          viewBox={`0 0 1000 ${Math.max(CHART_HEIGHT, totalHeight)}`}
          width="100%"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Y-axis label (rotated) */}
          {chartData.y_label ? (
            <text
              fill="#4f6678"
              fontSize="13"
              textAnchor="middle"
              transform="rotate(-90 14 200)"
              x="14"
              y="200"
            >
              {chartData.y_label}
            </text>
          ) : null}

          {/* Bars */}
          {bars.map((bar, i) => {
            const y    = i * (ROW_HEIGHT + ROW_GAP) + 20;
            const t    = bars.length > 1 ? i / (bars.length - 1) : 0;
            const fill = interpolateColor(t);
            const widthPct = (bar.value / maxValue);
            const barW = (1000 - LEFT_LABEL_WIDTH - RIGHT_PADDING) * widthPct;
            return (
              <g key={`${bar.label}-${i}`}>
                <text
                  dominantBaseline="middle"
                  fill="#274459"
                  fontSize="12"
                  textAnchor="end"
                  x={LEFT_LABEL_WIDTH - 8}
                  y={y + ROW_HEIGHT / 2}
                >
                  {bar.label}
                </text>
                <rect
                  fill={fill}
                  height={ROW_HEIGHT}
                  rx="4"
                  ry="4"
                  width={barW}
                  x={LEFT_LABEL_WIDTH}
                  y={y}
                />
                <text
                  dominantBaseline="middle"
                  fill="#1d2935"
                  fontSize="11"
                  x={LEFT_LABEL_WIDTH + barW + 6}
                  y={y + ROW_HEIGHT / 2}
                >
                  {Number.isInteger(bar.value)
                    ? bar.value.toLocaleString()
                    : bar.value.toFixed(4)}
                </text>
              </g>
            );
          })}

          {/* X-axis label */}
          {chartData.x_label ? (
            <text
              fill="#4f6678"
              fontSize="13"
              textAnchor="middle"
              x={(1000 + LEFT_LABEL_WIDTH) / 2}
              y={Math.max(CHART_HEIGHT, totalHeight) - 6}
            >
              {chartData.x_label}
            </text>
          ) : null}
        </svg>
      </div>
    </div>
  );
}

export default ChartView;

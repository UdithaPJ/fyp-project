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

function makeBfsHistogramChart(rawResult) {
  const inner = rawResult?.result || {};
  const distances = inner.distances || [];
  if (!Array.isArray(distances) || distances.length === 0) return null;

  const reachable = distances
    .map((d) => Number(d))
    .filter((d) => Number.isFinite(d) && d >= 0);
  if (reachable.length === 0) return null;

  const maxD = Math.max(...reachable);
  const counts = new Array(maxD + 1).fill(0);
  for (const d of reachable) {
    counts[d] += 1;
  }

  return {
    chart_type: "bar",
    labels: counts.map((_, i) => String(i)),
    values: counts,
    title: "Distance distribution (BFS)",
    x_label: "Number of nodes",
    y_label: "Distance level",
  };
}

function ChartView({ algorithmName, chartData, rawResult }) {
  const derivedChart = useMemo(() => {
    if (algorithmName !== "bfs") return null;
    return makeBfsHistogramChart(rawResult);
  }, [algorithmName, rawResult]);

  const effectiveChart = chartData && chartData.chart_type !== "none"
    ? chartData
    : derivedChart;

  const isUnsupported =
    !effectiveChart ||
    effectiveChart.unsupported ||
    effectiveChart.chart_type === "none" ||
    !Array.isArray(effectiveChart.labels) ||
    effectiveChart.labels.length === 0;

  const bars = useMemo(() => {
    if (isUnsupported) return [];
    const pairs = (effectiveChart.labels || []).map((label, i) => ({
      label,
      value: Number(effectiveChart.values?.[i] ?? 0),
      value2: Array.isArray(effectiveChart.series_auth)
        ? Number(effectiveChart.series_auth?.[i] ?? 0)
        : null,
    }));
    if (algorithmName === "bfs") {
      pairs.sort((a, b) => Number(a.label) - Number(b.label));
    } else {
      pairs.sort((a, b) => b.value - a.value);
    }
    return pairs.slice(0, MAX_BARS);
  }, [effectiveChart, isUnsupported, algorithmName]);

  if (isUnsupported) {
    return (
      <div className="viz-empty">
        Chart view is not available for this algorithm.
      </div>
    );
  }

  const hasSecondSeries = bars.some((b) => b.value2 !== null);

  const maxValue = Math.max(
    ...bars.map((b) => Math.max(b.value, b.value2 ?? 0)),
    1e-9,
  );
  const totalHeight = bars.length * (ROW_HEIGHT + ROW_GAP) + 60;
  const heightStr = `${Math.max(CHART_HEIGHT, totalHeight)}px`;

  const series1Label = effectiveChart?.y_label || "Score";
  const series2Label = "Authority score";
  const series2Color = "#d95f43";

  return (
    <div className="chart-view-wrap">
      {effectiveChart.title ? (
        <h3 className="chart-title">{effectiveChart.title}</h3>
      ) : null}

      {hasSecondSeries ? (
        <div className="chart-legend" role="list" aria-label="Chart legend">
          <span className="chart-legend-item" role="listitem">
            <span className="chart-legend-swatch" style={{ background: interpolateColor(0.7) }} />
            {series1Label}
          </span>
          <span className="chart-legend-item" role="listitem">
            <span className="chart-legend-swatch" style={{ background: series2Color }} />
            {series2Label}
          </span>
        </div>
      ) : null}

      <div className="chart-svg-wrap" style={{ height: heightStr }}>
        <svg
          height="100%"
          preserveAspectRatio="none"
          viewBox={`0 0 1000 ${Math.max(CHART_HEIGHT, totalHeight)}`}
          width="100%"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* Y-axis label (rotated) */}
          {effectiveChart.y_label ? (
            <text
              fill="#4f6678"
              fontSize="13"
              textAnchor="middle"
              transform="rotate(-90 14 200)"
              x="14"
              y="200"
            >
              {effectiveChart.y_label}
            </text>
          ) : null}

          {/* Bars */}
          {bars.map((bar, i) => {
            const y    = i * (ROW_HEIGHT + ROW_GAP) + 20;
            const t    = bars.length > 1 ? i / (bars.length - 1) : 0;
            const fill = interpolateColor(t);
            const widthPct = (bar.value / maxValue);
            const barW = (1000 - LEFT_LABEL_WIDTH - RIGHT_PADDING) * widthPct;
            const bar2W = bar.value2 !== null
              ? (1000 - LEFT_LABEL_WIDTH - RIGHT_PADDING) * ((bar.value2 || 0) / maxValue)
              : 0;

            const subH = hasSecondSeries ? (ROW_HEIGHT - 4) / 2 : ROW_HEIGHT;
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

                {/* Series 1 */}
                <rect
                  fill={fill}
                  height={subH}
                  rx="4"
                  ry="4"
                  width={barW}
                  x={LEFT_LABEL_WIDTH}
                  y={y + (hasSecondSeries ? 0 : 0)}
                />
                <text
                  dominantBaseline="middle"
                  fill="#1d2935"
                  fontSize="11"
                  x={LEFT_LABEL_WIDTH + barW + 6}
                  y={y + (hasSecondSeries ? subH / 2 : ROW_HEIGHT / 2)}
                >
                  {Number.isInteger(bar.value)
                    ? bar.value.toLocaleString()
                    : bar.value.toFixed(4)}
                </text>

                {/* Series 2 (HITS authority) */}
                {hasSecondSeries ? (
                  <>
                    <rect
                      fill={series2Color}
                      height={subH}
                      rx="4"
                      ry="4"
                      width={bar2W}
                      x={LEFT_LABEL_WIDTH}
                      y={y + subH + 4}
                      opacity="0.9"
                    />
                    <text
                      dominantBaseline="middle"
                      fill="#1d2935"
                      fontSize="11"
                      x={LEFT_LABEL_WIDTH + bar2W + 6}
                      y={y + subH + 4 + subH / 2}
                    >
                      {Number.isInteger(bar.value2 || 0)
                        ? Number(bar.value2 || 0).toLocaleString()
                        : Number(bar.value2 || 0).toFixed(4)}
                    </text>
                  </>
                ) : null}
              </g>
            );
          })}

          {/* X-axis label */}
          {effectiveChart.x_label ? (
            <text
              fill="#4f6678"
              fontSize="13"
              textAnchor="middle"
              x={(1000 + LEFT_LABEL_WIDTH) / 2}
              y={Math.max(CHART_HEIGHT, totalHeight) - 6}
            >
              {effectiveChart.x_label}
            </text>
          ) : null}
        </svg>
      </div>
    </div>
  );
}

export default ChartView;

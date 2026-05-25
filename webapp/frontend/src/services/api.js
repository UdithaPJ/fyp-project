const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function parseResponse(response) {
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || "Request failed.");
  }

  return payload;
}

export async function uploadDataset(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  return parseResponse(response);
}

export async function detectColumns(payload) {
  const response = await fetch(`${API_BASE_URL}/detect`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseResponse(response);
}

export async function preprocessDataset(payload) {
  const response = await fetch(`${API_BASE_URL}/preprocess`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseResponse(response);
}

function parseEventLine(line) {
  if (!line.trim()) {
    return null;
  }

  try {
    return JSON.parse(line);
  } catch {
    return null;
  }
}

export async function preprocessDatasetWithProgress(payload, onProgress) {
  const response = await fetch(`${API_BASE_URL}/preprocess/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error("Unable to start preprocessing.");
  }

  if (!response.body) {
    throw new Error("Streaming is not supported in this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const event = parseEventLine(line);
      if (!event) {
        continue;
      }

      if (event.event === "progress") {
        onProgress?.({
          step: event.step || "processing",
          percentage: Number(event.percentage || 0),
        });
      } else if (event.event === "error") {
        throw new Error(event.detail || "Preprocessing failed.");
      } else if (event.event === "result") {
        result = event.data;
      }
    }

    if (done) {
      break;
    }
  }

  const trailingEvent = parseEventLine(buffer);
  if (trailingEvent?.event === "error") {
    throw new Error(trailingEvent.detail || "Preprocessing failed.");
  }
  if (trailingEvent?.event === "result") {
    result = trailingEvent.data;
  }

  if (!result) {
    throw new Error("Preprocessing completed without a result.");
  }

  return result;
}

// ===========================================================================
// Algorithm execution (Steps 4-7)
// ===========================================================================

export async function getGraphStats(uploadId) {
  const response = await fetch(
    `${API_BASE_URL}/graph/stats/${encodeURIComponent(uploadId)}`,
  );
  return parseResponse(response);
}

export async function getAlgorithmCatalog() {
  const response = await fetch(`${API_BASE_URL}/algorithms/catalog`);
  return parseResponse(response);
}

export async function runAlgorithm(uploadId, algorithm, mode, params) {
  const response = await fetch(`${API_BASE_URL}/algorithms/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      upload_id: uploadId,
      algorithm,
      mode,
      params: params || {},
    }),
  });
  return parseResponse(response);
}

export async function getJobStatus(jobId) {
  const response = await fetch(
    `${API_BASE_URL}/algorithms/status/${encodeURIComponent(jobId)}`,
  );
  return parseResponse(response);
}

export async function getResults(jobId) {
  const response = await fetch(
    `${API_BASE_URL}/results/${encodeURIComponent(jobId)}`,
  );
  return parseResponse(response);
}

/**
 * Stream NDJSON progress events from the algorithm runner.
 *
 * Same implementation pattern as preprocessDatasetWithProgress — opens a
 * streaming GET, parses one JSON object per line, and dispatches:
 *   {type: "progress"} → onEvent
 *   {type: "result"}   → onDone (terminal)
 *   {type: "error"}    → onError (terminal)
 */
export async function streamAlgorithmProgress(jobId, onEvent, onDone, onError) {
  const response = await fetch(
    `${API_BASE_URL}/algorithms/run/stream/${encodeURIComponent(jobId)}`,
  );

  if (!response.ok) {
    onError?.(new Error("Unable to start algorithm stream."));
    return;
  }
  if (!response.body) {
    onError?.(new Error("Streaming is not supported in this browser."));
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  function dispatch(event) {
    if (!event || !event.type) {
      return;
    }
    if (event.type === "progress") {
      onEvent?.({
        stage: event.stage || "running",
        percent: Number(event.percent || 0),
        message: event.message || "",
      });
    } else if (event.type === "result") {
      onDone?.(event.data);
    } else if (event.type === "error") {
      onError?.(new Error(event.message || "Algorithm failed."));
    }
  }

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      dispatch(parseEventLine(line));
    }

    if (done) {
      break;
    }
  }

  const trailing = parseEventLine(buffer);
  if (trailing) {
    dispatch(trailing);
  }
}

/**
 * Trigger a browser download for a backend file response.
 * Used by both CSV and JSON export helpers.
 */
async function triggerFileDownload(url, fallbackName) {
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "Export failed.");
  }

  // Try to extract a filename from the Content-Disposition header
  const disposition = response.headers.get("content-disposition") || "";
  const match = disposition.match(/filename="?([^"]+)"?/i);
  const filename = match ? match[1] : fallbackName;

  const blob = await response.blob();
  const objectUrl = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = objectUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(objectUrl);
}

export async function exportResultsCSV(jobId) {
  await triggerFileDownload(
    `${API_BASE_URL}/results/${encodeURIComponent(jobId)}/export/csv`,
    `result_${jobId.slice(0, 8)}.csv`,
  );
}

export async function exportResultsJSON(jobId) {
  await triggerFileDownload(
    `${API_BASE_URL}/results/${encodeURIComponent(jobId)}/export/json`,
    `result_${jobId.slice(0, 8)}.json`,
  );
}

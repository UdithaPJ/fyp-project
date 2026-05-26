import { useEffect, useState } from "react";
import AlgorithmSelector from "./components/AlgorithmSelector";
import ExportPanel from "./components/ExportPanel";
import GraphSummary from "./components/GraphSummary";
import Mapping from "./components/Mapping";
import ResultsView from "./components/ResultsView";
import RunAnalysis from "./components/RunAnalysis";
import Upload from "./components/Upload";
import Validation from "./components/Validation";
import { detectColumns, getGpuStatus, preprocessDatasetWithProgress } from "./services/api";

const STEP_TITLES = [
  "Upload Dataset",
  "Map Columns",
  "Validate Data",
  "Graph Summary",
  "Algorithm Setup",
  "Run Analysis",
  "Results",
  "Export",
];

function App() {
  const [step, setStep] = useState(0);
  const [error, setError] = useState("");
  const [gpuStatus, setGpuStatus] = useState(null);   // null = checking

  // Check GPU availability once on mount — before the user does anything.
  useEffect(() => {
    getGpuStatus()
      .then(setGpuStatus)
      .catch(() =>
        setGpuStatus({
          available: false,
          device_name: null,
          cuda_version: null,
          driver: "none",
          error: "Could not reach the backend to check GPU status.",
        }),
      );
  }, []);
  const [uploadResult, setUploadResult] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [mapping, setMapping] = useState({
    source: "",
    target: "",
    weight: "",
  });
  const [duplicateMethod, setDuplicateMethod] = useState("mean");
  const [preprocessResult, setPreprocessResult] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [isPreprocessing, setIsPreprocessing] = useState(false);
  const [preprocessProgress, setPreprocessProgress] = useState({
    step: "",
    percentage: 0,
  });
  // Step 4-7 shared state
  const [algorithmConfig, setAlgorithmConfig] = useState(null);
  const [jobId, setJobId]                     = useState(null);
  const [algorithmResult, setAlgorithmResult] = useState(null);

  useEffect(() => {
    if (!uploadResult) {
      return;
    }

    let isActive = true;

    async function loadSuggestions() {
      setIsDetecting(true);
      setError("");
      try {
        const result = await detectColumns({
          columns: uploadResult.columns,
          sample: uploadResult.preview,
        });
        if (!isActive) {
          return;
        }

        setDetectionResult(result);
        setMapping({
          source: result.source || "",
          target: result.target || "",
          weight: result.weight || "",
        });
      } catch (requestError) {
        if (!isActive) {
          return;
        }
        setError(requestError.message);
      } finally {
        if (isActive) {
          setIsDetecting(false);
        }
      }
    }

    loadSuggestions();

    return () => {
      isActive = false;
    };
  }, [uploadResult]);

  useEffect(() => {
    setPreprocessResult(null);
    setPreprocessProgress({ step: "", percentage: 0 });
  }, [mapping.source, mapping.target, mapping.weight, duplicateMethod]);

  async function handleUploadSuccess(result) {
    setUploadResult(result);
    setDetectionResult(null);
    setDuplicateMethod("mean");
  }

  async function handleRunPreprocess() {
    setIsPreprocessing(true);
    setError("");
    setPreprocessResult(null);
    setPreprocessProgress({
      step: "starting",
      percentage: 0,
    });
    try {
      const result = await preprocessDatasetWithProgress(
        {
          upload_id: uploadResult.upload_id,
          mapping: {
            source: mapping.source,
            target: mapping.target,
            weight: mapping.weight || null,
          },
          duplicate_method: duplicateMethod,
        },
        (progress) => {
          setPreprocessProgress({
            step: progress.step,
            percentage: progress.percentage,
          });
        },
      );
      setPreprocessResult(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsPreprocessing(false);
    }
  }

  function handleReset() {
    setStep(0);
    setError("");
    setUploadResult(null);
    setDetectionResult(null);
    setMapping({ source: "", target: "", weight: "" });
    setDuplicateMethod("mean");
    setPreprocessResult(null);
    setIsDetecting(false);
    setIsPreprocessing(false);
    setPreprocessProgress({ step: "", percentage: 0 });
    setAlgorithmConfig(null);
    setJobId(null);
    setAlgorithmResult(null);
  }

  function handleAlgorithmConfigured(config) {
    setAlgorithmConfig(config);
    setJobId(null);
    setAlgorithmResult(null);
    setStep(5);
  }

  function handleRunComplete(newJobId, resultData) {
    setJobId(newJobId);
    setAlgorithmResult(resultData);
    setStep(6);
  }

  function handleRunAnother() {
    setAlgorithmConfig(null);
    setJobId(null);
    setAlgorithmResult(null);
    setStep(4);
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Local Graph Preprocessing</p>
          <h1>Framework</h1>
          <p className="subtitle">
            Upload a file, confirm the column mapping, validate the cleaned
            output, and build graph data entirely.
          </p>
        </div>
        <button className="secondary-button" onClick={handleReset} type="button">
          Start Over
        </button>
      </header>

      <section className="stepper">
        {STEP_TITLES.map((title, index) => (
          <div
            className={`step ${index === step ? "current" : ""} ${index < step ? "done" : ""}`}
            key={title}
          >
            <span>{index + 1}</span>
            <strong>{title}</strong>
          </div>
        ))}
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      {/* GPU availability gate — shown on the upload step while checking,
          and as a persistent blocking error if no CUDA GPU is detected.    */}
      {gpuStatus === null ? (
        <div className="gpu-status-checking">
          Checking for CUDA-capable GPU…
        </div>
      ) : !gpuStatus.available ? (
        <div className="gpu-status-error">
          <strong>No NVIDIA CUDA GPU detected</strong>
          <p>
            This framework requires a CUDA-capable NVIDIA GPU.
            All analyses run exclusively on the GPU.
          </p>
          {gpuStatus.error ? (
            <p className="gpu-status-detail">{gpuStatus.error}</p>
          ) : null}
          <p>
            Please run this application on a machine with a supported NVIDIA
            GPU and the appropriate CUDA drivers installed, then refresh the
            page.
          </p>
        </div>
      ) : step === 0 ? (
        <div className="gpu-status-ok">
          GPU detected: <strong>{gpuStatus.device_name}</strong>
          {gpuStatus.cuda_version ? ` — CUDA ${gpuStatus.cuda_version}` : ""}
        </div>
      ) : null}

      <main className={`panel${gpuStatus && !gpuStatus.available ? " panel--disabled" : ""}`}>
        {step === 0 && gpuStatus && !gpuStatus.available ? null : step === 0 ? (
          <Upload
            uploadResult={uploadResult}
            isDetecting={isDetecting}
            onUploadSuccess={handleUploadSuccess}
            onContinue={() => setStep(1)}
          />
        ) : null}

        {step === 1 ? (
          <Mapping
            columns={uploadResult?.columns || []}
            detection={detectionResult}
            mapping={mapping}
            onChange={setMapping}
            onBack={() => setStep(0)}
            onNext={() => setStep(2)}
            isDetecting={isDetecting}
          />
        ) : null}

        {step === 2 ? (
          <Validation
            rowCount={uploadResult?.row_count || 0}
            duplicateMethod={duplicateMethod}
            onDuplicateMethodChange={setDuplicateMethod}
            onBack={() => setStep(1)}
            onRunPreprocess={handleRunPreprocess}
            onContinue={() => setStep(3)}
            validation={preprocessResult?.validation || null}
            isPreprocessing={isPreprocessing}
            hasValidMapping={Boolean(mapping.source && mapping.target)}
            progress={preprocessProgress}
          />
        ) : null}

        {step === 3 ? (
          <>
            <GraphSummary
              preprocessResult={preprocessResult}
              onBack={() => setStep(2)}
            />
            <div className="action-row continue-row">
              <button
                className="primary-button"
                disabled={!preprocessResult}
                onClick={() => setStep(4)}
                type="button"
              >
                Continue to Algorithm Setup →
              </button>
            </div>
          </>
        ) : null}

        {step === 4 ? (
          <AlgorithmSelector
            onBack={() => setStep(3)}
            onNext={handleAlgorithmConfigured}
            uploadId={uploadResult?.upload_id}
          />
        ) : null}

        {step === 5 ? (
          <RunAnalysis
            algorithmConfig={algorithmConfig}
            onBack={() => setStep(4)}
            onComplete={handleRunComplete}
            uploadId={uploadResult?.upload_id}
          />
        ) : null}

        {step === 6 ? (
          <ResultsView
            algorithmName={algorithmConfig?.algorithm}
            jobId={jobId}
            onBack={() => setStep(5)}
            onContinue={() => setStep(7)}
            result={algorithmResult}
          />
        ) : null}

        {step === 7 ? (
          <ExportPanel
            algorithmName={algorithmConfig?.algorithm}
            jobId={jobId}
            onRunAnother={handleRunAnother}
            onStartOver={handleReset}
          />
        ) : null}
      </main>
    </div>
  );
}

export default App;

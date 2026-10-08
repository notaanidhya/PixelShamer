import React from "react";
import { ShieldCheck, AlertTriangle, AlertOctagon, HelpCircle, Eye, Cpu, Zap } from "lucide-react";
import { formatLocalTimestamp } from "../api/client";

export default function DeepfakePanel({ result, isAnalyzing }) {
  if (isAnalyzing) {
    return (
      <div className="workbench-panel diagnostics-panel" id="tour-diagnostics">
        <div className="panel-header">
          <div className="panel-title">
            <Cpu size={15} className="spin text-highlight" />
            <span>Neural Forensics in Progress...</span>
          </div>
        </div>
        <div className="analyzing-skeleton mono">
          <div className="skeleton-line-long"></div>
          <div className="skeleton-line-short"></div>
          <div className="skeleton-grid"></div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="workbench-panel diagnostics-panel" id="tour-diagnostics">
        <div className="panel-header">
          <div className="panel-title">
            <ShieldCheck size={15} />
            <span>Deepfake Forensics & Face Diagnostics</span>
          </div>
        </div>
        <div className="empty-diagnostics mono">
          <Cpu size={32} className="text-muted" />
          <span>Awaiting facial image. Load a sample benchmark or upload an image to execute forensics.</span>
        </div>
      </div>
    );
  }

  const fakePct = (result.fake_confidence * 100).toFixed(1);
  const verdict = result.verdict || "UNKNOWN";
  const stats = result.statistics || {};

  const getVerdictStyle = (v) => {
    switch (v) {
      case "AUTHENTIC":
        return {
          badgeClass: "badge-authentic",
          icon: <ShieldCheck size={18} className="text-emerald-400" />,
          label: "AUTHENTIC FACE",
          color: "var(--color-emerald, #10b981)",
          bg: "rgba(16, 185, 129, 0.12)",
          description: "Natural organic skin texture, coherent eye reflections, and absence of synthetic boundary seams."
        };
      case "SUSPICIOUS":
        return {
          badgeClass: "badge-suspicious",
          icon: <AlertTriangle size={18} className="text-amber-400" />,
          label: "SUSPICIOUS SYNTHESIS",
          color: "var(--color-amber, #f59e0b)",
          bg: "rgba(245, 158, 11, 0.12)",
          description: "Borderline facial frequency anomalies or boundary blending inconsistencies detected."
        };
      case "LIKELY_FAKE":
        return {
          badgeClass: "badge-fake",
          icon: <AlertOctagon size={18} className="text-rose-500" />,
          label: "LIKELY DEEPFAKE",
          color: "var(--color-rose, #ef4444)",
          bg: "rgba(239, 68, 68, 0.15)",
          description: "Strong neural synthesis fingerprints, boundary gradient anomalies, or GAN upsampling artifacts."
        };
      default:
        return {
          badgeClass: "badge-neutral",
          icon: <HelpCircle size={18} className="text-gray-400" />,
          label: "NO FACE LOCALIZED",
          color: "var(--text-muted, #94a3b8)",
          bg: "rgba(148, 163, 184, 0.1)",
          description: "No frontal human face was localized in the image. Please upload a clear face portrait."
        };
    }
  };

  const vInfo = getVerdictStyle(verdict);

  return (
    <div className="workbench-panel diagnostics-panel" id="tour-diagnostics">
      {/* Panel Header with Verdict Badge */}
      <div className="panel-header">
        <div className="panel-title">
          <ShieldCheck size={15} />
          <span>Deepfake Forensics & Face Diagnostics</span>
        </div>
        <div
          className="verdict-pill mono"
          style={{ color: vInfo.color, backgroundColor: vInfo.bg, borderColor: vInfo.color }}
        >
          {vInfo.icon}
          <span>{vInfo.label}</span>
        </div>
      </div>

      <div className="panel-body">
        {/* File & Timestamp Metadata with proper horizontal separation */}
        <div className="panel-meta-row mono">
          <span className="meta-filename" title={result.filename}>
            {result.filename}
          </span>
          <span className="meta-timestamp">
            {formatLocalTimestamp(result.processed_at)}
          </span>
        </div>

        {/* Hero Score Box */}
        <div className="score-hero-card" style={{ borderLeft: `4px solid ${vInfo.color}` }}>
          <div className="score-hero-top">
            <span className="mono text-xs text-muted tracking-wider uppercase">
              EFFICIENTNET-B5 NEURAL CONFIDENCE
            </span>
          </div>

          <div className="score-hero-number-row">
            <div className="score-hero-big mono" style={{ color: vInfo.color }}>
              {result.face_detected ? `${fakePct}%` : "—"}
            </div>
            <div className="score-hero-label-desc">
              <div className="score-hero-summary text-sm font-medium">
                {vInfo.description}
              </div>
              {result.face_detected && (
                <div className="score-bar-track" style={{ marginTop: "0.6rem" }}>
                  <div
                    className="score-bar-fill"
                    style={{
                      width: `${Math.min(100, Math.max(0, result.fake_confidence * 100))}%`,
                      backgroundColor: vInfo.color
                    }}
                  />
                </div>
              )}
            </div>
          </div>
        </div>

      {/* Face Localization Telemetry */}
      <div className="telemetry-card">
        <div className="card-header-compact">
          <Eye size={14} className="text-highlight" />
          <span className="mono text-xs text-muted uppercase tracking-wider">
            Face Localization & Preprocessing
          </span>
        </div>
        <div className="metrics-compact-grid mono text-xs">
          <div className="metric-chip">
            <span className="text-muted">Detection Status</span>
            <span className={result.face_detected ? "text-emerald-400 font-bold" : "text-rose-400 font-bold"}>
              {result.face_detected ? "LOCALIZED (100%)" : "NOT DETECTED"}
            </span>
          </div>
          {result.face_bbox && (
            <>
              <div className="metric-chip">
                <span className="text-muted">Face Crop Resolution</span>
                <span className="font-semibold text-highlight">
                  {result.face_bbox.w} × {result.face_bbox.h} px
                </span>
              </div>
              <div className="metric-chip">
                <span className="text-muted">Bounding Box Coordinates</span>
                <span className="text-muted">
                  X:{result.face_bbox.x}, Y:{result.face_bbox.y}
                </span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Spectral & Forensic Diagnostics */}
      {result.face_detected && stats && (
        <div className="telemetry-card">
          <div className="card-header-compact">
            <Cpu size={14} className="text-highlight" />
            <span className="mono text-xs text-muted uppercase tracking-wider">
              Frequency & Spectral Diagnostics
            </span>
          </div>
          <div className="metrics-compact-grid mono text-xs">
            <div className="metric-chip" title="Spectral energy outside 0.5x Nyquist frequency in 2D Fourier domain">
              <span className="text-muted">FFT HF Energy Ratio</span>
              <span className="font-bold text-highlight">
                {stats.fft_high_freq_ratio ?? "—"}
              </span>
            </div>
            <div className="metric-chip" title="Discontinuity across 8x8 DCT grid boundaries exposing compression cues">
              <span className="text-muted">DCT Blockiness</span>
              <span className="font-bold text-highlight">
                {stats.dct_blockiness ?? "—"}
              </span>
            </div>
            <div className="metric-chip" title="Laplacian variance measuring edge gradient sharpness">
              <span className="text-muted">Laplacian Variance</span>
              <span className="font-semibold">
                {stats.laplacian_variance ?? "—"}
              </span>
            </div>
            <div className="metric-chip" title="Reference-free Immerkär noise estimation">
              <span className="text-muted">Noise Sigma (Immerkär)</span>
              <span className="font-semibold">
                {stats.noise_sigma ?? "—"}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Spatial Explainability Callout */}
      <div className="explainability-callout mono text-xs">
        <Zap size={14} className="text-amber-400 shrink-0" />
        <p>
          <strong>Grad-CAM Spatial Explainability:</strong> Switch the viewport mode above to{" "}
          <span className="text-highlight">OVERLAY</span> or <span className="text-highlight">HEATMAP</span> to inspect which facial regions triggered the forgery activation score.
        </p>
      </div>
      </div>
    </div>
  );
}

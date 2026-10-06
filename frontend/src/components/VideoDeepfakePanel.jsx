import React from "react";
import { ShieldCheck, AlertTriangle, AlertOctagon, Film, Clock, Cpu, Zap, Eye } from "lucide-react";
import { formatLocalTimestamp } from "../api/client";

export default function VideoDeepfakePanel({ result, isAnalyzing }) {
  if (isAnalyzing) {
    return (
      <div className="workbench-panel diagnostics-panel" id="tour-diagnostics">
        <div className="panel-header">
          <div className="panel-title">
            <Cpu size={15} className="spin text-highlight" />
            <span>Spatio-Temporal Forensics in Progress...</span>
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
            <Film size={15} />
            <span>Video Deepfake Forensics</span>
          </div>
        </div>
        <div className="empty-diagnostics mono">
          <Film size={32} className="text-muted" />
          <span>Upload a video to execute spatio-temporal forensic analysis with Bi-LSTM temporal sequence scoring and Grad-CAM peak-frame explainability.</span>
        </div>
      </div>
    );
  }

  const fakePct = (result.fake_confidence * 100).toFixed(1);
  const verdict = result.verdict || "UNKNOWN";
  const timeline = result.timeline || [];

  const getVerdictStyle = (v) => {
    switch (v) {
      case "AUTHENTIC":
        return {
          badgeClass: "badge-authentic",
          icon: <ShieldCheck size={18} className="text-emerald-400" />,
          label: "AUTHENTIC VIDEO",
          color: "var(--color-emerald, #10b981)",
          bg: "rgba(16, 185, 129, 0.12)",
          description: "Continuous biomechanical consistency across all sampled frames. Natural temporal facial dynamics with coherent inter-frame transition gradients."
        };
      case "SUSPICIOUS":
        return {
          badgeClass: "badge-suspicious",
          icon: <AlertTriangle size={18} className="text-amber-400" />,
          label: "SUSPICIOUS SYNTHESIS",
          color: "var(--color-amber, #f59e0b)",
          bg: "rgba(245, 158, 11, 0.12)",
          description: "Borderline temporal anomalies detected. Unnatural inter-frame blending or subtle facial warping inconsistencies observed."
        };
      case "LIKELY_FAKE":
        return {
          badgeClass: "badge-fake",
          icon: <AlertOctagon size={18} className="text-rose-500" />,
          label: "LIKELY DEEPFAKE",
          color: "var(--color-rose, #ef4444)",
          bg: "rgba(239, 68, 68, 0.15)",
          description: "Definitive temporal manipulation detected. Severe biomechanical discontinuity, face-swap boundary artifacts, or re-enactment warping."
        };
      default:
        return {
          badgeClass: "badge-neutral",
          icon: <Film size={18} className="text-gray-400" />,
          label: "UNKNOWN",
          color: "var(--text-muted, #94a3b8)",
          bg: "rgba(148, 163, 184, 0.1)",
          description: "Analysis result unavailable."
        };
    }
  };

  const vInfo = getVerdictStyle(verdict);
  const peakFrame = timeline.find((f) => f.is_peak_anomaly);

  return (
    <div className="workbench-panel diagnostics-panel" id="tour-diagnostics">
      {/* Panel Header */}
      <div className="panel-header">
        <div className="panel-title">
          <Film size={15} />
          <span>Video Deepfake Forensics</span>
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
        {/* Filename & Timestamp */}
        <div className="panel-meta-row mono">
          <span className="meta-filename" title={result.filename}>
            {result.filename}
          </span>
          <span className="meta-timestamp">
            {formatLocalTimestamp(result.processed_at)}
          </span>
        </div>

        {/* Hero Confidence Score */}
        <div className="score-hero-card" style={{ borderLeft: `4px solid ${vInfo.color}` }}>
          <div className="score-hero-top">
            <span className="mono text-xs text-muted tracking-wider uppercase">
              Bi-LSTM Temporal Forgery Confidence
            </span>
          </div>
          <div className="score-hero-number-row">
            <div className="score-hero-big mono" style={{ color: vInfo.color }}>
              {fakePct}%
            </div>
            <div className="score-hero-label-desc">
              <div className="score-hero-summary text-sm font-medium">
                {vInfo.description}
              </div>
              <div className="score-bar-track" style={{ marginTop: "0.6rem" }}>
                <div
                  className="score-bar-fill"
                  style={{
                    width: `${Math.min(100, Math.max(0, result.fake_confidence * 100))}%`,
                    backgroundColor: vInfo.color
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Video Metadata */}
        <div className="telemetry-card">
          <div className="card-header-compact">
            <Clock size={14} className="text-highlight" />
            <span className="mono text-xs text-muted uppercase tracking-wider">
              Video Temporal Metadata
            </span>
          </div>
          <div className="metrics-compact-grid mono text-xs">
            <div className="metric-chip">
              <span className="text-muted">Video Duration</span>
              <span className="font-semibold text-highlight">{result.duration_seconds?.toFixed(2)}s</span>
            </div>
            <div className="metric-chip">
              <span className="text-muted">Frames Analyzed</span>
              <span className="font-semibold text-highlight">{result.total_frames_analyzed}</span>
            </div>
            <div className="metric-chip">
              <span className="text-muted">Decision Threshold</span>
              <span className="font-semibold">{(result.threshold * 100).toFixed(1)}%</span>
            </div>
            <div className="metric-chip">
              <span className="text-muted">Verdict</span>
              <span className="font-bold" style={{ color: vInfo.color }}>{verdict}</span>
            </div>
          </div>
        </div>

        {/* Peak Anomaly Frame */}
        {peakFrame && (
          <div className="telemetry-card">
            <div className="card-header-compact">
              <Zap size={14} className="text-highlight" />
              <span className="mono text-xs text-muted uppercase tracking-wider">
                Peak Anomaly Frame
              </span>
            </div>
            <div className="metrics-compact-grid mono text-xs">
              <div className="metric-chip">
                <span className="text-muted">Frame Index</span>
                <span className="font-bold" style={{ color: "var(--color-rose)" }}>
                  #{peakFrame.frame_index + 1} of {timeline.length}
                </span>
              </div>
              <div className="metric-chip">
                <span className="text-muted">Timestamp</span>
                <span className="font-semibold text-highlight">{result.peak_anomaly_timestamp?.toFixed(2)}s</span>
              </div>
              <div className="metric-chip">
                <span className="text-muted">Frame Forgery Score</span>
                <span className="font-bold" style={{ color: "var(--color-rose)" }}>
                  {(result.peak_anomaly_confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Forensic Narrative */}
        {result.analysis_summary && (
          <div className="telemetry-card">
            <div className="card-header-compact">
              <ShieldCheck size={14} className="text-highlight" />
              <span className="mono text-xs text-muted uppercase tracking-wider">
                Forensic Narrative
              </span>
            </div>
            <p className="mono text-xs" style={{ lineHeight: "1.6", color: "var(--text-secondary)" }}>
              {result.analysis_summary}
            </p>
          </div>
        )}

        {/* Grad-CAM Callout */}
        <div className="explainability-callout mono text-xs">
          <Eye size={14} className="text-amber-400 shrink-0" />
          <p>
            <strong>Grad-CAM Peak Frame:</strong> Click the{" "}
            <span className="text-highlight">PEAK ANOMALY</span> frame bar in the timeline below the video to seek to it, then click{" "}
            <span className="text-highlight">View Grad-CAM Heatmap</span> to see which facial regions triggered the forgery score.
          </p>
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect, useRef } from "react";
import {
  Layers,
  Sliders,
  Activity,
  ChevronRight,
  ChevronLeft,
  X,
  Maximize2,
  Terminal,
  Zap,
  ShieldCheck,
  UploadCloud,
  History
} from "lucide-react";
import "./WalkthroughTour.css";

const TOUR_STEPS = [
  {
    id: "introduction",
    isIntro: true,
    targetId: null, // Center modal
    placement: "center",
    icon: <ShieldCheck size={16} className="text-highlight" />,
    title: "PROJECT INTRODUCTION",
    subtitle: "PixelShamer • Unified AI Media Forensics & Deepfake Detection",
    description: (
      <>
        Welcome to <strong>PixelShamer</strong>, an industrial-grade AI media forensics and computer vision system. The platform resolves a fundamental security and inspection challenge: <strong>distinguishing between synthetic generative manipulations</strong> (deepfake face swaps, neural synthesis, manipulated video) and <strong>physical optical degradations</strong> (defocus blur, noise, exposure flaws, surface defects) in both images and video.
      </>
    ),
    features: [
      "Deepfake Face & Video Forensics: Detects manipulated portraits and video deepfakes using EfficientNet and Spatio-Temporal Bi-LSTM models with Grad-CAM heatmaps",
      "Physical Quality & Defect Triage: Evaluates 22 deterministic computer vision metrics, computes an objective 0–100 quality score, and localizes defects via ConvAutoencoder",
      "Unified Dual-Pipeline Workbench: Switch seamlessly between Deepfake Detection and Image Quality inspection with instant benchmark testing",
      "100% Air-Gapped & Private: Fast local CPU/GPU inference with zero telemetry, external API calls, or cloud dependencies"
    ],
    actionLabel: "Explore App Steps",
  },
  {
    id: "presets",
    targetId: "tour-preset-chips",
    placement: "bottom",
    icon: <Sliders size={15} className="text-highlight" />,
    title: "BENCHMARK PRESET CHIPS",
    subtitle: "Instant Testing Without File Uploads",
    description: (
      <>
        Test the system immediately! Click any of the <strong>curated benchmark presets</strong> to evaluate specific physical degradation families (clean, blur, noise, exposure, defects) or deepfake forensics (authentic portraits vs. face-swapped deepfakes).
      </>
    ),
    features: [
      "Zero-latency demonstration of all degradation and deepfake families",
      "Evaluates continuous scoring and issue classification live",
      "Pre-loaded from curated benchmark inspection suites"
    ],
    actionLabel: "Try Pristine Clean",
    onAction: (helpers) => {
      helpers.loadPresetSample("clean");
    },
  },
  {
    id: "dropzone",
    targetId: "tour-dropzone",
    placement: "bottom",
    icon: <UploadCloud size={15} className="text-highlight" />,
    title: "STREAM & FILE INGESTION",
    subtitle: "Drag & Drop, Browse, or Clipboard Paste",
    description: (
      <>
        Upload your own inspection media via <strong>drag-and-drop</strong>, file browsing, or by pressing <strong>Ctrl+V</strong> to paste images directly from your clipboard.
      </>
    ),
    features: [
      "Images: JPEG, PNG, WEBP, and BMP formats (up to 15 MB)",
      "Videos: MP4, AVI, MOV, MKV, and WEBM clips (up to 100 MB)",
      "Strict PIL/OpenCV header integrity verification with zero disk leakage"
    ],
  },
  {
    id: "viewport",
    targetId: "tour-viewport",
    placement: "right",
    icon: <Maximize2 size={15} className="text-highlight" />,
    title: "SPATIAL & TEMPORAL COMPARATOR",
    subtitle: "Original, Jet Colormap & Timeline Scrubber",
    description: (
      <>
        Inspect anomalies with spatial and temporal precision. Toggle between:
        <br />
        • <strong>Original</strong>: Full-resolution raw media capture.
        <br />
        • <strong>Overlay / Heatmap</strong>: Jet colormap Grad-CAM forgery localization or autoencoder defect residual.
        <br />
        • <strong>Video Timeline</strong>: Interactive 16-keyframe scrubber pinpoints peak anomaly timestamps.
      </>
    ),
    features: [
      "Jet colormap: Red = Anomaly peak / blend seam, Blue = Nominal surface",
      "Sub-millisecond peak anomaly frame synchronization for video clips",
      "Full-screen magnification and opacity blend control"
    ],
  },
  {
    id: "diagnostics",
    targetId: "tour-diagnostics",
    placement: "left",
    icon: <Activity size={15} className="text-highlight" />,
    title: "FORENSIC TRIAGE & SCORING",
    subtitle: "Deepfake Confidence & Calibrated Quality",
    description: (
      <>
        Review immediate diagnostic verdicts tailored to your active pipeline:
        <br />
        • <strong>Deepfake Mode</strong>: Forgery Probability (0–100%), Authentic vs. Fake verdict, face crop confidence, and manipulation indicators.
        <br />
        • <strong>Quality Mode</strong>: 0–100 Quality Index with PCHIP monotonic spline calibration and multi-label issue severity triage.
      </>
    ),
    features: [
      "Clear verdict triage: AUTHENTIC / DEEPFAKE or ACCEPTABLE / DEGRADED / DEFECTIVE",
      "Diminishing-returns compounding for multi-issue co-occurrences",
      "Peak temporal frame timestamp and facial bounding coordinates"
    ],
  },
  {
    id: "metrics",
    targetId: "tour-metrics-matrix",
    placement: "top",
    icon: <Layers size={15} className="text-highlight" />,
    title: "22-METRIC TELEMETRY MATRIX",
    subtitle: "Deterministic Computer Vision Diagnostics",
    description: (
      <>
        Examine deep signal processing metrics grouped into 7 mathematical families: <strong>Sharpness</strong> (Laplacian, FFT), <strong>Exposure</strong>, <strong>Contrast</strong>, <strong>Immerkær Noise</strong>, <strong>GLCM Texture</strong>, and <strong>DCT Blockiness</strong>, alongside deepfake face crop telemetry.
      </>
    ),
    features: [
      "Exposes exact numerical indicators for explainable inspection",
      "Reference-free noise sigma estimation (ISO 15739 standard)",
      "Expandable family cards with normal ranges and physical units"
    ],
  },
  {
    id: "history",
    targetId: "tour-history-nav",
    placement: "bottom",
    icon: <History size={15} className="text-highlight" />,
    title: "AUDIT TRAIL & PRIVACY",
    subtitle: "Session Isolation & Global Feed",
    description: (
      <>
        Every analysis is logged in the database. Switch to <strong>Audit History</strong> to review previous inspections. <strong>My Session</strong> keeps your tests private, while <strong>Global Feed</strong> shows public benchmark runs.
      </>
    ),
    features: [
      "Full pagination with sorting and status filtering across all modalities",
      "Inspect detailed diagnostic snapshots for past image and video records",
      "Exportable audit logs for forensics and quality compliance"
    ],
    actionLabel: "Finish Walkthrough",
  },
];

export default function WalkthroughTour({
  isOpen,
  onClose,
  onSwitchTab,
  onLoadPreset,
  hasActiveResult,
}) {
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [targetRect, setTargetRect] = useState(null);
  const [popoverPos, setPopoverPos] = useState({ top: "50%", left: "50%", transform: "translate(-50%, -50%)" });
  const popoverRef = useRef(null);

  const step = TOUR_STEPS[currentStepIdx];
  const isFirstStep = currentStepIdx === 0;
  const isLastStep = currentStepIdx === TOUR_STEPS.length - 1;

  const totalAppSteps = TOUR_STEPS.filter((s) => !s.isIntro).length;
  const appStepNumber = step.isIntro
    ? 0
    : TOUR_STEPS.slice(0, currentStepIdx + 1).filter((s) => !s.isIntro).length;

  // Calculate target element rect & position popover
  const updatePosition = () => {
    if (!isOpen || !step) return;

    if (!step.targetId) {
      setTargetRect(null);
      setPopoverPos({
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        position: "fixed",
      });
      return;
    }

    const el = document.getElementById(step.targetId);
    if (!el) {
      setTargetRect(null);
      setPopoverPos({
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        position: "fixed",
      });
      return;
    }

    // Scroll target into view
    el.scrollIntoView({ behavior: "smooth", block: "center" });

    const rect = el.getBoundingClientRect();
    const padding = 8;

    setTargetRect({
      top: rect.top - padding,
      left: rect.left - padding,
      width: rect.width + padding * 2,
      height: rect.height + padding * 2,
    });

    const popoverWidth = step.isIntro ? 460 : 420;
    const popoverHeight = step.isIntro ? 420 : 350;
    const placement = step.placement || "bottom";

    // Desktop 2-column layout checks (no overlap)
    if (window.innerWidth >= 960) {
      if (placement === "left") {
        // Place strictly to the LEFT of the target panel (e.g. over viewport)
        let left = rect.left - popoverWidth - 24;
        if (left < 16) left = 16;
        let top = Math.max(16, Math.min(window.innerHeight - popoverHeight - 16, rect.top + 20));
        setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
        return;
      } else if (placement === "right") {
        // Place strictly to the RIGHT of the target panel (e.g. over diagnostics)
        let left = rect.right + 24;
        if (left + popoverWidth > window.innerWidth - 16) left = window.innerWidth - popoverWidth - 16;
        let top = Math.max(16, Math.min(window.innerHeight - popoverHeight - 16, rect.top + 20));
        setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
        return;
      } else if (placement === "top") {
        // Place strictly ABOVE the target panel (e.g. above metrics matrix)
        let top = rect.top - popoverHeight - 24;
        if (top < 16) top = 16;
        let left = Math.max(16, Math.min(window.innerWidth - popoverWidth - 16, rect.left + (rect.width - popoverWidth) / 2));
        setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
        return;
      } else if (placement === "bottom") {
        // Place strictly BELOW the target panel (e.g. below presets)
        let top = rect.bottom + 20;
        if (top + popoverHeight > window.innerHeight - 16) top = window.innerHeight - popoverHeight - 16;
        let left = Math.max(16, Math.min(window.innerWidth - popoverWidth - 16, rect.left + (rect.width - popoverWidth) / 2));
        setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
        return;
      }
    }

    // Small screen / mobile fallback
    let top = rect.bottom + 16;
    if (top + popoverHeight > window.innerHeight - 16) {
      top = Math.max(16, rect.top - popoverHeight - 16);
    }
    let left = Math.max(16, (window.innerWidth - popoverWidth) / 2);
    setPopoverPos({ top: `${top}px`, left: `${left}px`, transform: "none", position: "fixed" });
  };

  useEffect(() => {
    if (!isOpen) return;

    // Auto-load Pristine Clean preset if advancing past Welcome without any active result
    if (currentStepIdx >= 1 && !hasActiveResult) {
      onLoadPreset?.("clean");
    }

    // Handle view tabs during tour
    if (step.id === "history") {
      onSwitchTab?.("workspace"); // Keep in workspace to highlight history button
    }

    const timer = setTimeout(updatePosition, 120);
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);

    return () => {
      clearTimeout(timer);
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
    };
  }, [currentStepIdx, isOpen, hasActiveResult]);

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        handleClose();
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        handleNext();
      } else if (e.key === "ArrowLeft") {
        handlePrev();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, currentStepIdx, hasActiveResult]);

  const handleNext = () => {
    if (!hasActiveResult) {
      onLoadPreset?.("clean");
    }
    if (isLastStep) {
      handleClose();
    } else {
      setCurrentStepIdx((prev) => prev + 1);
    }
  };

  const handlePrev = () => {
    if (!isFirstStep) {
      setCurrentStepIdx((prev) => prev - 1);
    }
  };

  const handleClose = () => {
    try {
      localStorage.setItem("pixelshamer_tour_seen_v2", "true");
    } catch (e) {}
    setCurrentStepIdx(0);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="tour-overlay-container">
      {/* SVG Backdrop with smooth spotlight cutout */}
      <svg className="tour-backdrop-svg">
        <defs>
          <mask id="tour-spotlight-mask">
            <rect x="0" y="0" width="100vw" height="100vh" fill="white" />
            {targetRect && (
              <rect
                x={targetRect.left}
                y={targetRect.top}
                width={targetRect.width}
                height={targetRect.height}
                rx="6"
                ry="6"
                fill="black"
              />
            )}
          </mask>
        </defs>
        <rect
          x="0"
          y="0"
          width="100vw"
          height="100vh"
          className="tour-mask-bg"
          mask="url(#tour-spotlight-mask)"
        />
      </svg>

      {/* Target outline pulse ring */}
      {targetRect && (
        <div
          className="tour-spotlight-pulse"
          style={{
            top: `${targetRect.top}px`,
            left: `${targetRect.left}px`,
            width: `${targetRect.width}px`,
            height: `${targetRect.height}px`,
          }}
        />
      )}

      {/* Popover Card */}
      <div
        ref={popoverRef}
        className={`tour-popover ${step.isIntro ? "is-intro" : ""}`}
        style={popoverPos}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="tour-popover-header">
          <div className={`tour-step-badge mono ${step.isIntro ? "intro-badge" : ""}`}>
            <span className={`tour-step-dot ${step.isIntro ? "intro-dot" : ""}`}></span>
            {step.isIntro ? (
              <span>INTRODUCTION • PROJECT OVERVIEW</span>
            ) : (
              <span>STEP {String(appStepNumber).padStart(2, "0")} / {String(totalAppSteps).padStart(2, "0")} • HOW TO USE</span>
            )}
          </div>
          <button
            className="tour-close-btn"
            onClick={handleClose}
            title="Exit Walkthrough (Esc)"
          >
            <X size={15} />
          </button>
        </div>

        {/* Title & Subtitle */}
        <div className="tour-title mono">
          {step.icon || <Terminal size={15} className="text-highlight" />}
          <span>{step.title}</span>
        </div>
        {step.subtitle && (
          <div className="text-muted mono" style={{ fontSize: "0.75rem", marginBottom: "0.6rem" }}>
            {step.subtitle}
          </div>
        )}

        {/* Description */}
        <div className="tour-description">{step.description}</div>

        {/* Feature Highlights */}
        {step.features && (
          <ul className="tour-feature-list">
            {step.features.map((feat, i) => (
              <li key={i} className="tour-feature-item">
                <span className="tour-feature-bullet">▸</span>
                <span>{feat}</span>
              </li>
            ))}
          </ul>
        )}

        {/* Interactive Action Box (if step has an action) */}
        {step.onAction && (
          <div className="tour-action-box mono">
            <span className="tour-action-text">Interactive Demonstration:</span>
            <button
              className="btn btn-secondary btn-sm"
              onClick={() => step.onAction({ loadPresetSample: onLoadPreset })}
              style={{ fontSize: "0.75rem", padding: "0.25rem 0.6rem" }}
            >
              <Zap size={12} className="text-highlight" />
              <span>{step.actionLabel || "Run Action"}</span>
            </button>
          </div>
        )}

        {/* Footer Navigation */}
        <div className="tour-popover-footer">
          {/* Step Progress Pills */}
          <div className="tour-progress-bar">
            {TOUR_STEPS.map((s, i) => (
              <div
                key={i}
                className={`tour-progress-pill ${
                  s.isIntro ? "pill-intro" : ""
                } ${
                  i === currentStepIdx
                    ? "active"
                    : i < currentStepIdx
                    ? "completed"
                    : ""
                }`}
                onClick={() => setCurrentStepIdx(i)}
                style={{ cursor: "pointer" }}
                title={s.isIntro ? "Introduction: Project Overview" : `Step ${i}: ${s.title}`}
              />
            ))}
          </div>

          {/* Nav Buttons */}
          <div className="tour-nav-buttons mono">
            {!isFirstStep && (
              <button
                className="btn btn-secondary btn-sm"
                onClick={handlePrev}
              >
                <ChevronLeft size={13} />
                <span>Prev</span>
              </button>
            )}
            <button
              className="btn btn-primary btn-sm"
              onClick={handleNext}
            >
              <span>{isLastStep ? "Finish" : isFirstStep ? "Explore Steps" : "Next"}</span>
              <ChevronRight size={13} />
            </button>
          </div>
        </div>

        {/* Hotkey hint */}
        <div className="tour-hotkey-hint mono">
          <span><kbd className="tour-kbd">Esc</kbd> Exit</span>
          <span>•</span>
          <span><kbd className="tour-kbd">→</kbd> Next</span>
          <span>•</span>
          <span><kbd className="tour-kbd">←</kbd> Prev</span>
        </div>
      </div>
    </div>
  );
}

import React, { useEffect, useState } from "react";
import { Activity, Layers, History, ShieldCheck, RefreshCw, HelpCircle } from "lucide-react";
import { checkHealth } from "../api/client";

export default function Header({ activeTab, onTabChange, onStartTour, activePipeline = "quality", onPipelineChange }) {
  const [health, setHealth] = useState({ status: "checking", models_loaded: false, deepfake_models_loaded: false, latency: null });

  const fetchHealth = async () => {
    try {
      const data = await checkHealth();
      setHealth({
        status: data.status === "ok" ? "online" : "degraded",
        models_loaded: data.models_loaded,
        deepfake_models_loaded: data.deepfake_models_loaded,
        latency: data.latency,
      });
    } catch (err) {
      setHealth({ status: "offline", models_loaded: false, deepfake_models_loaded: false, latency: null });
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 25000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="site-header">
      <div className="header-inner">
        {/* Brand identity */}
        <div className="brand-section">
          <div className="brand-logo">
            <Layers size={18} className="text-highlight" />
          </div>
          <div className="brand-title">
            Pixel<span className="brand-sub">Shamer</span>
          </div>
        </div>

        {/* Forensic Pipeline Mode Switcher */}
        <div className="pipeline-switcher mono">
          <button
            type="button"
            className={`pipeline-btn ${activePipeline === "quality" ? "active" : ""}`}
            onClick={() => onPipelineChange("quality")}
            title="Switch to Image Quality Assessment & Surface Defect Localization"
          >
            <Activity size={14} />
            <span>Quality & Defect</span>
          </button>
          <button
            type="button"
            className={`pipeline-btn ${(activePipeline === "deepfake" || activePipeline === "video_deepfake") ? "active" : ""}`}
            onClick={() => onPipelineChange("deepfake")}
            title="Switch to Deepfake Face Forgery Detection & Grad-CAM Analysis"
          >
            <ShieldCheck size={14} />
            <span>Deepfake Detection</span>
          </button>
        </div>

        {/* View Switcher Tabs */}
        <nav className="header-nav mono" id="tour-nav-tabs">
          <button
            className={`nav-tab ${activeTab === "workspace" ? "active" : ""}`}
            onClick={() => onTabChange("workspace")}
            id="tour-workspace-nav"
          >
            <Activity size={14} />
            <span>Workbench</span>
          </button>
          <button
            className={`nav-tab ${activeTab === "history" ? "active" : ""}`}
            onClick={() => onTabChange("history")}
            id="tour-history-nav"
          >
            <History size={14} />
            <span>Audit History</span>
          </button>
        </nav>

        {/* Guided Tour Trigger Button */}
        <button
          className="header-tour-btn mono tour-launch-btn"
          onClick={onStartTour}
          title="Launch Interactive Guided Tour"
        >
          <HelpCircle size={14} />
          <span>Guided Tour</span>
        </button>

        {/* System Telemetry & Health */}
        <div className="header-status">
          {health.status === "online" ? (
            <div className="telemetry-badge online mono">
              <span className="status-dot online"></span>
              <span>API ONLINE</span>
              {health.latency && <span className="latency">({health.latency}ms)</span>}
              {health.models_loaded && (
                <span className="models-tag">
                  <ShieldCheck size={12} /> MODELS READY
                </span>
              )}
            </div>
          ) : health.status === "checking" ? (
            <div className="telemetry-badge checking mono">
              <RefreshCw size={12} className="spin" />
              <span>CONNECTING...</span>
            </div>
          ) : (
            <div className="telemetry-badge offline mono">
              <span className="status-dot offline"></span>
              <span>BACKEND OFFLINE</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

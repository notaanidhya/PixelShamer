import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import UploadZone from "./components/UploadZone";
import ImageViewer from "./components/ImageViewer";
import VideoTimelineViewer from "./components/VideoTimelineViewer";
import DiagnosticsPanel from "./components/DiagnosticsPanel";
import DeepfakePanel from "./components/DeepfakePanel";
import VideoDeepfakePanel from "./components/VideoDeepfakePanel";
import MetricsMatrix from "./components/MetricsMatrix";
import HistoryTable from "./components/HistoryTable";
import WalkthroughTour from "./components/WalkthroughTour";
import {
  analyzeImage,
  analyzeDeepfake,
  analyzeVideoDeepfake,
  getPresetAnalysis,
  getFileMediaType,
} from "./api/client";
import { AlertCircle, X } from "lucide-react";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("workspace");
  const [activePipeline, setActivePipeline] = useState("quality"); // "quality" | "deepfake"
  const [activeResult, setActiveResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isTourOpen, setIsTourOpen] = useState(false);
  const [mediaType, setMediaType] = useState("image"); // "image" | "video"
  const [uploadProgress, setUploadProgress] = useState(null);

  // Auto-launch walkthrough for first-time visitors
  useEffect(() => {
    try {
      const hasSeenTour = localStorage.getItem("pixelshamer_tour_seen");
      if (!hasSeenTour) {
        const timer = setTimeout(() => {
          setIsTourOpen(true);
        }, 800);
        return () => clearTimeout(timer);
      }
    } catch (e) {}
  }, []);

  const handlePipelineChange = (pipeline) => {
    if (pipeline === activePipeline) return;
    setActivePipeline(pipeline);
    setActiveResult(null);
    setPreviewUrl(null);
    setErrorMsg(null);
    setMediaType("image");
    setUploadProgress(null);
  };

  const handleFileSelected = async (file) => {
    if (!file) return;
    setErrorMsg(null);
    setActiveResult(null);
    setUploadProgress(null);

    // Detect media type
    const detectedType = getFileMediaType(file);
    setMediaType(detectedType);

    // For video: create a local object URL so the video player shows immediately
    const localUrl = URL.createObjectURL(file);
    setPreviewUrl(localUrl);
    setIsAnalyzing(true);

    try {
      let data;

      if (detectedType === "video") {
        // Video deepfake analysis (only makes sense in deepfake pipeline)
        if (activePipeline !== "deepfake") {
          throw new Error("Video files can only be analyzed in the Deepfake Detection pipeline. Switch the pipeline in the header.");
        }
        data = await analyzeVideoDeepfake(file, (pct) => setUploadProgress(pct));
      } else if (activePipeline === "deepfake") {
        // Image deepfake analysis
        data = await analyzeDeepfake(file);
      } else {
        // Image quality analysis
        data = await analyzeImage(file);
      }

      setActiveResult(data);
    } catch (err) {
      console.error("Analysis failed:", err);
      const detail = err.response?.data?.detail || err.message || "Failed to analyze file.";
      setErrorMsg(detail);
    } finally {
      setIsAnalyzing(false);
      setUploadProgress(null);
    }
  };

  const handlePresetSelected = async (preset) => {
    if (!preset) return;
    setErrorMsg(null);
    setActiveResult(null);
    setPreviewUrl(preset.file);
    setMediaType("image"); // All presets are images
    setIsAnalyzing(true);
    setUploadProgress(null);

    try {
      if (activePipeline === "deepfake") {
        // Deepfake preset: fetch sample blob and execute deepfake analysis
        const response = await fetch(preset.file);
        if (!response.ok) throw new Error(`Preset file not found: ${preset.file}`);
        const blob = await response.blob();
        const filename = preset.file.split("/").pop();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
        const data = await analyzeDeepfake(file);
        setActiveResult(data);
      } else {
        // Quality preset: 1. Fast path: load pre-computed telemetry directly from DB
        try {
          const data = await getPresetAnalysis(preset.id);
          setActiveResult(data);
        } catch (err) {
          console.warn("Fast preset fetch failed, falling back to direct analysis...", err);
          // Fallback: fetch blob and run standard quality analysis
          const response = await fetch(preset.file);
          const blob = await response.blob();
          const filename = preset.file.split("/").pop();
          const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
          const data = await analyzeImage(file);
          setActiveResult(data);
        }
      }
    } catch (err) {
      console.error("Analysis failed:", err);
      const detail = err.response?.data?.detail || err.message || "Failed to load preset.";
      setErrorMsg(detail);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTourLoadPreset = async (presetId = "defect") => {
    const presetFileMap = {
      clean: { id: "clean", file: "/samples/sample_pristine___clean.jpg" },
      defect: { id: "defect", file: "/samples/sample_synthetic_defect.jpg" },
      blur: { id: "blur", file: "/samples/sample_blur_defocus.jpg" },
      noise: { id: "noise", file: "/samples/sample_gaussian_noise.jpg" },
    };
    const targetPreset = presetFileMap[presetId] || presetFileMap["defect"];
    handlePresetSelected(targetPreset);
  };

  // Determine which viewer and panel to show
  const isVideoResult = mediaType === "video";
  const showVideoPanel = activePipeline === "deepfake" && isVideoResult;
  const showImageDeepfakePanel = activePipeline === "deepfake" && !isVideoResult;
  const showQualityPanel = activePipeline === "quality";

  return (
    <div className="app-container">
      {/* Header with Navigation and Live Health Status */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        activePipeline={activePipeline}
        onPipelineChange={handlePipelineChange}
        onStartTour={() => setIsTourOpen(true)}
      />

      <main className="main-content">
        {/* Global Error Banner */}
        {errorMsg && (
          <div className="error-banner mono">
            <div className="error-content">
              <AlertCircle size={16} className="text-status-defective" />
              <span>{errorMsg}</span>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => setErrorMsg(null)}>
              <X size={14} />
            </button>
          </div>
        )}

        {/* View Switcher */}
        {activeTab === "workspace" ? (
          <div className="workspace-view">
            {/* Upload Zone — accepts images and (in deepfake mode) videos */}
            <UploadZone
              onFileSelected={handleFileSelected}
              onPresetSelected={handlePresetSelected}
              isAnalyzing={isAnalyzing}
              activePipeline={activePipeline}
            />

            {/* Split Screen Dual Workspace */}
            <div className="grid-2col workspace-grid">
              {/* Left Column: Spatial Viewport or Video Player */}
              <div className="workspace-col-left">
                {showVideoPanel ? (
                  <VideoTimelineViewer
                    result={activeResult}
                    previewUrl={previewUrl}
                    isAnalyzing={isAnalyzing}
                    uploadProgress={uploadProgress}
                  />
                ) : (
                  <ImageViewer
                    result={activeResult}
                    previewUrl={previewUrl}
                    isAnalyzing={isAnalyzing}
                  />
                )}
              </div>

              {/* Right Column: Telemetry Panel */}
              <div className="workspace-col-right">
                {showVideoPanel ? (
                  <VideoDeepfakePanel result={activeResult} isAnalyzing={isAnalyzing} />
                ) : showImageDeepfakePanel ? (
                  <DeepfakePanel result={activeResult} isAnalyzing={isAnalyzing} />
                ) : (
                  <DiagnosticsPanel result={activeResult} isAnalyzing={isAnalyzing} />
                )}
              </div>
            </div>

            {/* Bottom: Metrics Matrix (image results only — video has its own timeline) */}
            {!isVideoResult && activeResult && activeResult.statistics && (
              <div className="metrics-matrix-wrapper">
                <MetricsMatrix
                  statistics={activeResult.statistics}
                  pipeline={activePipeline}
                  result={activeResult}
                />
              </div>
            )}
          </div>
        ) : (
          <HistoryTable activePipeline={activePipeline} onPipelineChange={setActivePipeline} />
        )}
      </main>

      {/* Interactive Walkthrough Tour */}
      <WalkthroughTour
        isOpen={isTourOpen}
        onClose={() => setIsTourOpen(false)}
        onSwitchTab={setActiveTab}
        onLoadPreset={handleTourLoadPreset}
        hasActiveResult={!!activeResult}
      />
    </div>
  );
}

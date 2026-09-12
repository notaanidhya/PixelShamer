import React, { useState, useEffect } from "react";
import { Eye, Layers, Sliders } from "lucide-react";
import { getAssetUrl, formatLocalTimestamp } from "../api/client";

export default function ImageViewer({ result, previewUrl, isAnalyzing }) {
  const [viewMode, setViewMode] = useState("overlay"); // "original", "heatmap", "overlay"
  const [overlayOpacity, setOverlayOpacity] = useState(70);
  const [imageMeta, setImageMeta] = useState({ width: null, height: null });

  // Reset metadata when previewUrl or image changes
  useEffect(() => {
    setImageMeta({ width: null, height: null });
  }, [previewUrl]);

  // While analyzing, strictly suppress the previous image's heatmap overlay
  const originalSrc = previewUrl || (result?.image_url ? getAssetUrl(result.image_url) : null);
  const heatmapSrc = (!isAnalyzing && result?.heatmap_url) ? getAssetUrl(result.heatmap_url) : null;

  // Read natural dimensions
  const handleImageLoaded = (e) => {
    setImageMeta({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight,
    });
  };

  return (
    <div className="workbench-panel image-viewer-panel" id="tour-viewport">
      <div className="panel-header">
        <div className="panel-title">
          <Layers size={15} />
          <span>Spatial Inspection Viewport</span>
        </div>

        {/* View mode toggle tabs (Geometric / Monospace - No generic pills) */}
        {heatmapSrc && !isAnalyzing && (
          <div className="view-mode-tabs mono">
            <button
              className={`mode-tab ${viewMode === "original" ? "active" : ""}`}
              onClick={() => setViewMode("original")}
            >
              Original
            </button>
            <button
              className={`mode-tab ${viewMode === "overlay" ? "active" : ""}`}
              onClick={() => setViewMode("overlay")}
            >
              Overlay ({overlayOpacity}%)
            </button>
            <button
              className={`mode-tab ${viewMode === "heatmap" ? "active" : ""}`}
              onClick={() => setViewMode("heatmap")}
            >
              Raw Heatmap
            </button>
          </div>
        )}
      </div>

      <div className="image-viewport-container">
        {originalSrc ? (
          <div className="image-canvas-wrapper">
            {/* Base Image with explicit key so React remounts on source change */}
            <img
              key={originalSrc}
              src={viewMode === "heatmap" && heatmapSrc && !isAnalyzing ? heatmapSrc : originalSrc}
              alt="Inspection Target"
              className="canvas-image base-layer"
              onLoad={handleImageLoaded}
            />

            {/* Overlay Layer (Shown strictly when not analyzing and heatmap is available) */}
            {viewMode === "overlay" && heatmapSrc && !isAnalyzing && (
              <img
                key={`overlay-${heatmapSrc}`}
                src={heatmapSrc}
                alt="Anomaly Heatmap Overlay"
                className="canvas-image overlay-layer"
                style={{ opacity: overlayOpacity / 100 }}
              />
            )}

            {/* Active Scanning Animation while model processes the new image */}
            {isAnalyzing && (
              <div className="viewport-scanning-overlay">
                <div className="scanning-beam"></div>
                <div className="scanning-pill mono">
                  <span className="pulse-dot"></span>
                  <span>ANALYZING IMAGE TELEMETRY...</span>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="empty-viewport mono">
            <Eye size={36} className="text-muted" />
            <span>Select or drop an image above to begin diagnostic inspection.</span>
          </div>
        )}
      </div>

      {/* Viewport Footer Telemetry & Controls */}
      {originalSrc && (
        <div className="viewport-footer">
          {/* Metadata telemetry */}
          <div className="viewport-meta mono">
            <span className="meta-item">
              <span className="text-muted">FILE:</span>{" "}
              {isAnalyzing
                ? "Processing Frame..."
                : (result?.filename || previewUrl?.split("/").pop() || "Input Stream")}
            </span>
            {imageMeta.width && (
              <span className="meta-item">
                <span className="text-muted">RES:</span> {imageMeta.width}×{imageMeta.height} px
              </span>
            )}
            {!isAnalyzing && result?.processed_at && (
              <span className="meta-item">
                <span className="text-muted">ANALYZED:</span>{" "}
                {formatLocalTimestamp(result.processed_at, "time")}
              </span>
            )}
          </div>

          {/* Opacity slider for Overlay Mode */}
          {heatmapSrc && !isAnalyzing && viewMode === "overlay" && (
            <div className="opacity-slider-control mono">
              <Sliders size={13} className="text-secondary" />
              <span>Heatmap Blend:</span>
              <input
                type="range"
                min="10"
                max="100"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                className="opacity-slider"
              />
              <span className="slider-value">{overlayOpacity}%</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

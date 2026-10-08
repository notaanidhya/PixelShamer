import React, { useState, useRef, useEffect, useCallback } from "react";
import { Play, Pause, Film, Eye, Zap, Layers } from "lucide-react";
import { getAssetUrl, formatLocalTimestamp } from "../api/client";

/**
 * VideoTimelineViewer
 * Shows the uploaded video with an interactive 16-frame anomaly timeline beneath it.
 * Clicking any timeline bar seeks the video to that frame's timestamp.
 * Shows the Grad-CAM heatmap when the peak anomaly frame is selected.
 */
export default function VideoTimelineViewer({ result, previewUrl, isAnalyzing, uploadProgress }) {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [selectedFrameIdx, setSelectedFrameIdx] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);

  const videoSrc = result?.video_url ? getAssetUrl(result.video_url) : previewUrl;
  const heatmapSrc = result?.heatmap_url ? getAssetUrl(result.heatmap_url) : null;
  const timeline = result?.timeline || [];
  const peakIdx = timeline.findIndex((f) => f.is_peak_anomaly);

  // Reset on new result
  useEffect(() => {
    setSelectedFrameIdx(null);
    setShowHeatmap(false);
    setIsPlaying(false);
    setCurrentTime(0);
  }, [result?.id]);

  // Auto-select peak anomaly frame when result arrives
  useEffect(() => {
    if (peakIdx >= 0 && selectedFrameIdx === null) {
      setSelectedFrameIdx(peakIdx);
    }
  }, [peakIdx, selectedFrameIdx]);

  const handlePlayPause = () => {
    if (!videoRef.current) return;
    if (videoRef.current.paused) {
      const playPromise = videoRef.current.play();
      if (playPromise !== undefined) {
        playPromise
          .then(() => setIsPlaying(true))
          .catch((err) => {
            console.warn("Video play aborted or failed:", err);
            setIsPlaying(false);
          });
      }
    } else {
      videoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleTimeUpdate = () => {
    if (!videoRef.current) return;
    setCurrentTime(videoRef.current.currentTime);
  };

  const handleLoadedMetadata = () => {
    if (!videoRef.current) return;
    setDuration(videoRef.current.duration);
  };

  const handleFrameClick = useCallback((frame, idx) => {
    setSelectedFrameIdx(idx);
    if (videoRef.current && frame.timestamp_sec != null) {
      videoRef.current.currentTime = frame.timestamp_sec;
      setCurrentTime(frame.timestamp_sec);
    }
  }, []);

  const formatTime = (s) => {
    if (!s && s !== 0) return "—";
    const min = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1).padStart(4, "0");
    return `${min}:${sec}`;
  };

  const getBarColor = (prob) => {
    if (prob >= 0.65) return "var(--color-rose, #ef4444)";
    if (prob >= 0.40) return "var(--color-amber, #f59e0b)";
    return "var(--color-emerald, #10b981)";
  };

  const selectedFrame = selectedFrameIdx !== null ? timeline[selectedFrameIdx] : null;

  return (
    <div className="workbench-panel image-viewer-panel" id="tour-viewport">
      <div className="panel-header">
        <div className="panel-title">
          <Film size={15} />
          <span>Video Forensic Inspection Viewport</span>
        </div>

        {/* Heatmap toggle — only when a peak frame with heatmap is selected */}
        {heatmapSrc && selectedFrameIdx === peakIdx && !isAnalyzing && (
          <div className="view-mode-tabs mono">
            <button
              className={`mode-tab ${!showHeatmap ? "active" : ""}`}
              onClick={() => setShowHeatmap(false)}
            >
              <Play size={11} /> Video
            </button>
            <button
              className={`mode-tab ${showHeatmap ? "active" : ""}`}
              onClick={() => setShowHeatmap(true)}
            >
              <Eye size={11} /> Peak Frame Heatmap
            </button>
          </div>
        )}
      </div>

      <div className="image-viewport-container">
        {isAnalyzing ? (
          /* Uploading / Analyzing state */
          <div className="empty-viewport mono" style={{ flexDirection: "column", gap: "1rem" }}>
            <Film size={36} className="text-muted" style={{ animation: "pulse 1.5s infinite" }} />
            {uploadProgress != null && uploadProgress < 100 ? (
              <>
                <span>Uploading video... {uploadProgress}%</span>
                <div style={{ width: "200px", height: "4px", background: "var(--border-color)", borderRadius: "2px" }}>
                  <div style={{ width: `${uploadProgress}%`, height: "100%", background: "var(--color-highlight)", borderRadius: "2px", transition: "width 0.3s" }} />
                </div>
              </>
            ) : (
              <span className="text-highlight">Running Spatio-Temporal Neural Inference...</span>
            )}
          </div>
        ) : videoSrc && !showHeatmap ? (
          /* Video player */
          <div className="image-canvas-wrapper" style={{ position: "relative" }}>
            <video
              ref={videoRef}
              src={videoSrc}
              className="canvas-image base-layer"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onEnded={() => setIsPlaying(false)}
              style={{ cursor: "pointer", maxHeight: "320px", objectFit: "contain", background: "#000" }}
              onClick={handlePlayPause}
              playsInline
            />
            {/* Play/Pause overlay hint */}
            {!isPlaying && (
              <div
                style={{
                  position: "absolute", inset: 0, display: "flex",
                  alignItems: "center", justifyContent: "center",
                  background: "rgba(0,0,0,0.25)", cursor: "pointer",
                  borderRadius: "4px"
                }}
                onClick={handlePlayPause}
              >
                <div style={{
                  background: "rgba(0,0,0,0.6)", borderRadius: "50%",
                  width: "48px", height: "48px", display: "flex",
                  alignItems: "center", justifyContent: "center"
                }}>
                  <Play size={22} color="white" />
                </div>
              </div>
            )}
          </div>
        ) : videoSrc && showHeatmap && heatmapSrc ? (
          /* Grad-CAM heatmap of peak anomaly frame */
          <div className="image-canvas-wrapper">
            <img
              src={heatmapSrc}
              alt="Peak Anomaly Frame — Grad-CAM Heatmap"
              className="canvas-image base-layer"
              style={{ maxHeight: "320px", objectFit: "contain" }}
            />
          </div>
        ) : (
          <div className="empty-viewport mono">
            <Film size={36} className="text-muted" />
            <span>Drop a video or image above to begin forensic inspection.</span>
          </div>
        )}
      </div>

      {/* Frame Anomaly Timeline Chart */}
      {!isAnalyzing && timeline.length > 0 && (
        <div className="timeline-section">
          {/* Video playback bar + time */}
          {duration > 0 && (
            <div className="timeline-playback-row mono">
              <button className="timeline-playpause-btn" onClick={handlePlayPause}>
                {isPlaying ? <Pause size={12} /> : <Play size={12} />}
              </button>
              <div className="timeline-seek-track" onClick={(e) => {
                if (!videoRef.current) return;
                const rect = e.currentTarget.getBoundingClientRect();
                const ratio = (e.clientX - rect.left) / rect.width;
                videoRef.current.currentTime = ratio * duration;
              }}>
                <div
                  className="timeline-seek-fill"
                  style={{ width: `${(currentTime / duration) * 100}%` }}
                />
              </div>
              <span className="text-muted" style={{ fontSize: "10px", minWidth: "60px" }}>
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </div>
          )}

          {/* Frame bars */}
          <div className="timeline-chart-header mono">
            <Layers size={12} className="text-muted" />
            <span className="text-muted">Frame-by-Frame Anomaly Timeline — click to seek</span>
            {peakIdx >= 0 && (
              <span style={{ color: "var(--color-rose)", fontSize: "10px" }}>
                <Zap size={10} style={{ display: "inline", verticalAlign: "middle" }} /> Peak @ {formatTime(timeline[peakIdx]?.timestamp_sec)}
              </span>
            )}
          </div>
          <div className="timeline-bars-row">
            {timeline.map((frame, idx) => {
              const isSelected = selectedFrameIdx === idx;
              const isPeak = frame.is_peak_anomaly;
              const barH = Math.max(8, Math.round(frame.fake_probability * 60));
              const barColor = getBarColor(frame.fake_probability);

              return (
                <div
                  key={idx}
                  className={`timeline-bar-col ${isSelected ? "selected" : ""}`}
                  onClick={() => handleFrameClick(frame, idx)}
                  title={`Frame ${idx + 1} @ ${formatTime(frame.timestamp_sec)} — ${(frame.fake_probability * 100).toFixed(1)}% fake`}
                >
                  {isPeak && (
                    <span className="timeline-peak-marker">
                      <Zap size={9} color="var(--color-rose)" />
                    </span>
                  )}
                  <div
                    className={`timeline-bar ${isPeak ? "timeline-bar-peak" : ""} ${isSelected ? "timeline-bar-selected" : ""}`}
                    style={{
                      height: `${barH}px`,
                      background: barColor,
                      opacity: isSelected ? 1 : 0.65,
                      boxShadow: isPeak ? `0 0 6px ${barColor}` : "none"
                    }}
                  />
                  <span className="timeline-bar-label mono">{idx + 1}</span>
                </div>
              );
            })}
          </div>

          {/* Selected frame detail */}
          {selectedFrame && (
            <div className="timeline-frame-detail mono">
              <span>
                Frame {selectedFrame.frame_index + 1} — {formatTime(selectedFrame.timestamp_sec)} —{" "}
                <span style={{ color: getBarColor(selectedFrame.fake_probability), fontWeight: "bold" }}>
                  {(selectedFrame.fake_probability * 100).toFixed(1)}% forgery confidence
                </span>
                {selectedFrame.is_peak_anomaly && (
                  <span style={{ color: "var(--color-rose)", marginLeft: "8px" }}>
                    ⚡ PEAK ANOMALY
                  </span>
                )}
              </span>
              {heatmapSrc && selectedFrame.is_peak_anomaly && (
                <button
                  className="timeline-heatmap-btn mono"
                  onClick={() => setShowHeatmap((v) => !v)}
                >
                  {showHeatmap ? <Film size={11} /> : <Eye size={11} />}
                  {showHeatmap ? "Back to Video" : "View Grad-CAM Heatmap"}
                </button>
              )}
            </div>
          )}
        </div>
      )}

      {/* Footer metadata */}
      {(videoSrc || result) && !isAnalyzing && (
        <div className="viewport-footer">
          <div className="viewport-meta mono">
            <span className="meta-item">
              <span className="text-muted">FILE:</span>{" "}
              {result?.filename || "Video Input"}
            </span>
            {result?.duration_seconds > 0 && (
              <span className="meta-item">
                <span className="text-muted">DURATION:</span> {result.duration_seconds.toFixed(1)}s
              </span>
            )}
            {result?.total_frames_analyzed && (
              <span className="meta-item">
                <span className="text-muted">FRAMES:</span> {result.total_frames_analyzed} sampled
              </span>
            )}
            {result?.processed_at && (
              <span className="meta-item">
                <span className="text-muted">ANALYZED:</span>{" "}
                {formatLocalTimestamp(result.processed_at, "time")}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

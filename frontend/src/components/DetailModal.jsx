import React, { useEffect } from "react";
import { X, FileText } from "lucide-react";
import ImageViewer from "./ImageViewer";
import VideoTimelineViewer from "./VideoTimelineViewer";
import DiagnosticsPanel from "./DiagnosticsPanel";
import DeepfakePanel from "./DeepfakePanel";
import VideoDeepfakePanel from "./VideoDeepfakePanel";
import MetricsMatrix from "./MetricsMatrix";

export default function DetailModal({ item, onClose }) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  if (!item) return null;

  const isVideo = item.total_frames_analyzed !== undefined || !!item.video_url;
  const isDeepfakeImage = !isVideo && (item.fake_confidence !== undefined || item.face_detected !== undefined);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div className="modal-title mono">
            <FileText size={16} className="text-highlight" />
            <span>
              {isVideo ? "VIDEO AUDIT INSPECTION" : isDeepfakeImage ? "DEEPFAKE AUDIT INSPECTION" : "QUALITY AUDIT INSPECTION"}: {item.filename}
            </span>
            <span className="text-muted">(ID: #{item.id})</span>
          </div>
          <button className="btn btn-ghost modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        {/* Modal Content */}
        <div className="modal-body-scroll">
          <div className="grid-2col modal-grid">
            {isVideo ? (
              <>
                <VideoTimelineViewer result={item} previewUrl={null} isAnalyzing={false} />
                <VideoDeepfakePanel result={item} isAnalyzing={false} />
              </>
            ) : isDeepfakeImage ? (
              <>
                <ImageViewer result={item} previewUrl={null} isAnalyzing={false} />
                <DeepfakePanel result={item} isAnalyzing={false} />
              </>
            ) : (
              <>
                <ImageViewer result={item} previewUrl={null} isAnalyzing={false} />
                <DiagnosticsPanel result={item} isAnalyzing={false} />
              </>
            )}
          </div>

          {!isVideo && !isDeepfakeImage && item.statistics && (
            <div style={{ marginTop: "1.25rem" }}>
              <MetricsMatrix statistics={item.statistics} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

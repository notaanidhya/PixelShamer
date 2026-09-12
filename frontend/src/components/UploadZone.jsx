import React, { useState, useRef, useEffect } from "react";
import { UploadCloud, FileImage, Loader2 } from "lucide-react";

const QUALITY_PRESET_SAMPLES = [
  { id: "clean", label: "Pristine Clean", file: "/samples/sample_pristine___clean.jpg", tag: "NOMINAL" },
  { id: "blur", label: "Defocus Blur", file: "/samples/sample_blur_defocus.jpg", tag: "BLUR" },
  { id: "underexp", label: "Underexposure", file: "/samples/sample_underexposure.jpg", tag: "DARK" },
  { id: "overexp", label: "Overexposure", file: "/samples/sample_overexposure.jpg", tag: "BRIGHT" },
  { id: "noise", label: "Gaussian Noise", file: "/samples/sample_gaussian_noise.jpg", tag: "NOISE" },
  { id: "corrupt", label: "JPEG Glitch", file: "/samples/sample_jpeg_corruption.jpg", tag: "CORRUPT" },
  { id: "defect", label: "Physical Defect", file: "/samples/sample_synthetic_defect.jpg", tag: "DEFECT" },
  { id: "multi", label: "Multi-Degraded", file: "/samples/sample_multi-degradation.jpg", tag: "MULTI" },
];

const DEEPFAKE_PRESET_SAMPLES = [
  { id: "real_portrait_1", label: "Authentic Portrait 1", file: "/samples/deepfake/df_sample_real_portrait_1.jpg", tag: "REAL", isReal: true },
  { id: "real_portrait_2", label: "Authentic Portrait 2", file: "/samples/deepfake/df_sample_real_portrait_2.jpg", tag: "REAL", isReal: true },
  { id: "real_natural", label: "Authentic Natural", file: "/samples/deepfake/df_sample_real_natural.jpg", tag: "REAL", isReal: true },
  { id: "real_studio", label: "Authentic Studio", file: "/samples/deepfake/df_sample_real_studio.jpg", tag: "REAL", isReal: true },
  { id: "fake_easy", label: "Deepfake (Warped)", file: "/samples/deepfake/df_sample_fake_easy.jpg", tag: "FAKE", isReal: false },
  { id: "fake_gan", label: "Deepfake (GAN Synth)", file: "/samples/deepfake/df_sample_fake_gan.jpg", tag: "FAKE", isReal: false },
  { id: "fake_blend", label: "Deepfake (Boundary Blend)", file: "/samples/deepfake/df_sample_fake_blend.jpg", tag: "FAKE", isReal: false },
  { id: "fake_hard", label: "Deepfake (Subtle Swap)", file: "/samples/deepfake/df_sample_fake_hard.jpg", tag: "FAKE", isReal: false },
];

export default function UploadZone({ onFileSelected, onPresetSelected, isAnalyzing, activePipeline = "quality" }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [loadingPreset, setLoadingPreset] = useState(null);
  const fileInputRef = useRef(null);

  // Global Clipboard paste listener
  useEffect(() => {
    const handlePaste = (e) => {
      if (isAnalyzing) return;
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf("image") !== -1) {
          const blob = items[i].getAsFile();
          if (blob) {
            const pastedFile = new File([blob], `pasted_image_${Date.now()}.png`, { type: blob.type });
            onFileSelected(pastedFile);
            break;
          }
        }
      }
    };

    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [onFileSelected, isAnalyzing]);

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!isAnalyzing) setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (isAnalyzing) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      onFileSelected(file);
    }
  };

  const handlePresetClick = async (preset) => {
    if (isAnalyzing) return;
    try {
      setLoadingPreset(preset.id);
      if (onPresetSelected) {
        await onPresetSelected(preset);
      } else {
        const response = await fetch(preset.file);
        if (!response.ok) throw new Error("Preset file not accessible");
        const blob = await response.blob();
        const filename = preset.file.split("/").pop();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
        onFileSelected(file);
      }
    } catch (err) {
      console.error("Failed to load preset:", err);
    } finally {
      setLoadingPreset(null);
    }
  };

  return (
    <div className="upload-section">
      {/* Primary Drop Target */}
      <div
        id="tour-dropzone"
        className={`dropzone-box ${isDragOver ? "drag-active" : ""} ${isAnalyzing ? "analyzing-active" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isAnalyzing && fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          className="hidden-file-input"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              onFileSelected(e.target.files[0]);
            }
          }}
        />

        <div className="dropzone-content">
          <div className="dropzone-icon-well">
            {isAnalyzing ? (
              <Loader2 size={28} className="spin text-highlight" />
            ) : (
              <UploadCloud size={28} className="drop-icon" />
            )}
          </div>
          <div className="dropzone-text">
            <span className="primary-prompt">
              {isAnalyzing ? "Executing Hybrid Neural & Feature Inference..." : "Drop an image to inspect, or browse file"}
            </span>
            <span className="secondary-prompt mono">
              Supported: JPEG, PNG, WEBP, BMP • Max 15 MB • Direct Paste (Ctrl+V)
            </span>
          </div>
        </div>
      </div>

      {/* 1-Click Preset Samples Matrix */}
      <div className="preset-bar" id="tour-preset-chips">
        <div className="preset-header mono">
          <FileImage size={13} className="text-secondary" />
          <span>{activePipeline === "deepfake" ? "Deepfake Forensics Benchmarks (Real vs Manipulated):" : "Quick Quality & Defect Presets:"}</span>
        </div>
        <div className="preset-chips-grid">
          {(activePipeline === "deepfake" ? DEEPFAKE_PRESET_SAMPLES : QUALITY_PRESET_SAMPLES).map((preset) => (
            <button
              key={preset.id}
              className="preset-btn mono"
              disabled={isAnalyzing}
              onClick={() => handlePresetClick(preset)}
            >
              {loadingPreset === preset.id ? (
                <Loader2 size={12} className="spin" />
              ) : (
                <span
                  className={`preset-tag-indicator ${
                    activePipeline === "deepfake"
                      ? preset.isReal ? "tag-real" : "tag-fake"
                      : `tag-${preset.id}`
                  }`}
                ></span>
              )}
              <span className="preset-name">{preset.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

# Forensic Diagnostic Report: Deepfake False Positive Analysis
**Document Reference:** `DF-FA-2026-09-01`  
**Subject:** Investigation into 99% False Positive Deepfake Classification on Authentic Portraits  
**Target Architecture:** EfficientNet-B2 Binary Classifier + Grad-CAM Explainability + OpenCV Cascade Detection  

---

## 1. Executive Summary

Empirical testing of authentic user-captured camera images revealed a stark classification disparity:
* **Image A (`WIN_20260911_14_15_38_Pro.jpg`):** **2.9% Fake Confidence (`AUTHENTIC FACE`)** — Correctly classified.
* **Image B (`IMG20260708140549.jpg`):** **99.2% Fake Confidence (`LIKELY DEEPFAKE`)** — Severe False Positive.
* **Image C (`me.jpeg`):** **99.0% Fake Confidence (`LIKELY DEEPFAKE`)** — Severe False Positive.
* **Image D (`IMG-20260114-WA0216.jpg`):** **93.2% Fake Confidence (`LIKELY DEEPFAKE`)** — Severe False Positive.

Critically, **Image A and Image B are the exact same individual in similar indoor surroundings**. Controlled ablation and Grad-CAM spatial activation mapping demonstrate that the model is suffering from a classic computer vision failure mode: **Spurious Feature Correlation driven by Spectacles/Eyewear and High-Frequency Smartphone Edge Artifacts**.

---

## 2. Comparative Telemetry & Ablation Findings

| Parameter | Image A (Webcam) | Image B (Phone Camera) | Image C (`me.jpeg`) | Image D (WhatsApp) |
|---|:---:|:---:|:---:|:---:|
| **Subject** | Same Person | Same Person | Same Person | 3 Friends (User on Right) |
| **Eyeglasses Present?** | ❌ **No Glasses** |  **Wearing Glasses** |  **Wearing Glasses** |  **Right Person Wearing Glasses** |
| **Neural Logit** | **-1.062** | **+4.969** | **+4.065** | **+3.376** |
| **Fake Confidence** | **2.9%** | **99.2%** | **99.0%** | **93.2%** |
| **Final Verdict** | **`AUTHENTIC`** | **`LIKELY DEEPFAKE`** | **`LIKELY DEEPFAKE`** | **`LIKELY DEEPFAKE`** |
| **Face Crop Res** | $583 \times 583\text{ px}$ | $1839 \times 1839\text{ px}$ | $440 \times 440\text{ px}$ | $592 \times 592\text{ px}$ |
| **Laplacian Variance**| $152.77$ | $520.89$ | $58.13$ | $811.22$ |
| **Grad-CAM Hotspot** | Diffuse (Mouth/Jaw) | **Bullseye on Glasses/Bridge** | **Bullseye on Glasses/Bridge** | **Bullseye on Glasses/Bridge** |

### Controlled Region Ablation Experiment
To isolate the exact causal trigger for Image B (+4.969 logit), we performed progressive spatial masking:
1. **Full Image with Glasses:** Logit = **`+4.9686`** ($99.31\%$ fake)
2. **Eyes & Spectacles Region Blurred (Gaussian $51\times 51$):** Logit dropped to **`+2.3499`** ($91.29\%$ fake) — an immediate **$52.7\%$ logit reduction**.
3. **Mouth & Chin Only (Eyes & Nose excluded entirely):** Logit collapsed to **`+1.6922`** ($84.45\%$ fake).

---

## 3. Root Cause Decomposition

### 3.1 Training Dataset Bias: The CIPLab Generation Artifact
The fine-tuning dataset used for the prototype (`CIPLab Real & Fake Face Detection`) synthesizes deepfakes by digitally splicing facial components into real face canvases:
* **70.6% (678 / 960) of all fake images in CIPLab have artificial eye replacement.**
* When eyes are composited into a face canvas in synthetic datasets:
  1. Sharp boundary seams and rectangular gradient mismatches appear around the eye sockets and nose bridge.
  2. Specular reflection inconsistencies occur between the grafted eyes and ambient facial skin.
  3. Anti-aliasing or alpha-blending artifacts trace the rim of the manipulated region.

### 3.2 Eyewear Mimicry of Digital Splicing
Eyeglasses physically replicate almost every visual cue that the CNN learned to associate with spliced eye components:
* **High-Contrast Rigid Frames:** Dark or metallic rims cross the nasal bridge and encircle the orbits, mimicking boundary blend seams.
* **Lens Reflections & Anti-Reflective Coating:** Eyeglass lenses produce green/blue chromatic glare and specular highlights that break natural corneal reflection symmetry.
* **Refraction Discontinuity:** Lenses minifying or magnifying the facial outline behind the frame create an edge discontinuity that mirrors patch compositing.

### 3.3 Smartphone Image Signal Processor (ISP) Oversharpening
* `IMG20260708140549.jpg` is a massive **$14\text{ MP}$ ($3468 \times 4063$)** smartphone capture.
* Modern smartphone computational photography engines apply unsharp masking, high-contrast edge halos, and multi-frame HDR tone mapping.
* The face crop is **$1839 \times 1839$ pixels** with a Laplacian variance of **$520.89$**.
* When this hyper-sharpened frame is downsampled to $260 \times 260$ for EfficientNet-B2, edge gradients around fine features (wireframe glasses, eyelashes, skin pores) produce high-frequency aliasing patterns that the network misinterprets as GAN upsampling artifacts.

### 3.4 Single Point of Failure (Absence of Multi-Modal Gating)
Currently, `deepfake_inference.py` relies exclusively on the scalar sigmoid output of EfficientNet-B2:
$$\text{Verdict} = f(\sigma(\text{Logit}_{\text{EfficientNet}}))$$
Although the backend extracts 2D Fourier (FFT HF Ratio), DCT Blockiness, and Immerkär Noise Sigma, **these physical signals are currently display-only in the UI and do not participate in decision arbitration**. In genuine camera photos, natural photon noise is spatially uniform across the entire face; in deepfakes, the synthetic face patch possesses distinct frequency and noise signatures from the background.

---

## 4. Multi-Stage Remediation Strategy

### Phase A: Decision Fusion & Eyewear Heuristic Calibration (Immediate / No Retraining Needed)
1. **Ocular Feature Heatmap Gating:**
   * If Grad-CAM maximum activation is strictly concentrated ($>75\%$ energy) within the eye/spectacle horizontal band ($y \in [0.28, 0.55]$), but global spectral noise sigma is consistent with pristine camera sensors ($\sigma < 2.5$, FFT HF Ratio $< 0.015$), attenuate fake confidence into the `SUSPICIOUS` or `AUTHENTIC` tier.
2. **Anti-Aliasing Downsampler (Area Averaging):**
   * Replace standard linear interpolation with `cv2.INTER_AREA` combined with gentle pre-filtering for large camera captures ($>1000\text{px}$), mitigating smartphone ISP sharpening halos.
3. **Platt Scaling & Dual-Branch Verdict Fusion:**
   * Calibrate the raw neural logit against the deterministic frequency telemetry to prevent single-feature false alarms.

### Phase B: Eyewear-Augmented Dataset & Fine-Tuning (Medium-Term)
1. **Data Augmentation with Eyewear & Synthetic Glare:**
   * Integrate Albumentations `RandomSunFlare`, `CoarseDropout`, and synthetic glasses/reflection overlays onto authentic training samples. This explicitly forces EfficientNet to learn that high-contrast ocular frames exist on authentic human faces.
2. **Real-World Diverse Ingestion (FFHQ / CelebA-HQ):**
   * Supplement `training_real` with 1,000+ real portraits of diverse individuals wearing eyeglasses, reading glasses, and sunglasses across varied indoor/outdoor lighting conditions.
3. **Focal Loss / Hard Negative Mining:**
   * Mine the false-positive samples (`IMG20260708140549.jpg`, `me.jpeg`, etc.) as hard negatives during backpropagation to penalize weights that over-rely on nose-bridge and spectacle rims.

### Phase C: Temporal & Multi-Spectral Video Architecture (Phase 02 Target)
* As documented in the Phase 02 roadmap, single-frame spatial CNNs will be augmented with temporal sequence modeling (TCN/LSTM) and a dedicated 2D FFT magnitude spectrum CNN branch, eliminating reliance on static spatial boundary heuristics.

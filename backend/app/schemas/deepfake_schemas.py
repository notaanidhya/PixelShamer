"""
backend/app/schemas/deepfake_schemas.py
=======================================
Pydantic v2 schemas for Deepfake Detection requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class DeepfakeResponse(BaseModel):
    id: Optional[int] = None
    session_id: Optional[str] = None
    filename: str
    fake_confidence: float = Field(..., ge=0.0, le=1.0, description="Deepfake probability score [0.0, 1.0]")
    verdict: str = Field(..., description="AUTHENTIC, SUSPICIOUS, LIKELY_FAKE, or NOT_DETECTED")
    face_detected: bool = Field(True, description="Whether a face was localized in the image")
    face_bbox: Optional[Dict[str, int]] = Field(None, description="Bounding box {x, y, w, h} of detected face")
    image_url: Optional[str] = None
    heatmap_url: Optional[str] = None
    statistics: Optional[Dict[str, Any]] = None
    processed_at: Optional[datetime] = None

class PaginatedDeepfakeResponse(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[DeepfakeResponse]

class FrameTimelineItem(BaseModel):
    frame_index: int = Field(..., description="Index of extracted keyframe (0 to N-1)")
    timestamp_sec: float = Field(..., description="Timestamp in seconds from video start")
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Frame forgery probability [0.0, 1.0]")
    is_peak_anomaly: bool = Field(False, description="Whether this frame represents the highest forgery anomaly")

class VideoDeepfakeResponse(BaseModel):
    id: Optional[int] = None
    session_id: Optional[str] = None
    filename: str
    duration_seconds: float = Field(..., description="Total duration of video in seconds")
    total_frames_analyzed: int = Field(..., description="Number of uniform keyframes analyzed")
    fake_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall video forgery probability [0.0, 1.0]")
    verdict: str = Field(..., description="AUTHENTIC, SUSPICIOUS, or LIKELY_FAKE")
    threshold: float = Field(0.450, description="Calibrated decision threshold applied")
    peak_anomaly_timestamp: float = Field(..., description="Timestamp of highest anomaly frame in seconds")
    peak_anomaly_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of peak anomalous frame")
    timeline: List[FrameTimelineItem] = Field(..., description="Continuous frame-by-frame anomaly timeline")
    video_url: Optional[str] = None
    heatmap_url: Optional[str] = Field(None, description="Grad-CAM explainability heatmap on the peak anomalous frame")
    analysis_summary: str = Field(..., description="Forensic narrative summarizing video authenticity")
    processed_at: Optional[datetime] = None

class PaginatedVideoDeepfakeResponse(BaseModel):
    total: int
    page: int
    limit: int
    pages: int
    items: List[VideoDeepfakeResponse]

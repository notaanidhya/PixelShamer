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

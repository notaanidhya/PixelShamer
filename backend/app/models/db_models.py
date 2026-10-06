import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from backend.app.db.session import Base

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

class AnalysisRecord(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True, index=True)
    upload_time = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    quality_score = Column(Float, nullable=False)
    quality_label = Column(String(50), nullable=False, index=True)
    issues = Column(JSON, nullable=False, default=list)
    statistics = Column(JSON, nullable=False, default=dict)
    image_url = Column(String(500), nullable=False)
    heatmap_url = Column(String(500), nullable=True)

    def to_dict(self):
        iso_str = None
        if self.upload_time:
            dt = self.upload_time
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            iso_str = dt.isoformat()

        return {
            "id": self.id,
            "session_id": self.session_id,
            "filename": self.filename,
            "quality_score": round(self.quality_score, 1),
            "quality_label": self.quality_label,
            "issues": self.issues,
            "statistics": self.statistics,
            "image_url": self.image_url,
            "heatmap_url": self.heatmap_url,
            "processed_at": iso_str
        }

class DeepfakeRecord(Base):
    __tablename__ = "deepfake_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String(64), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True, index=True)
    upload_time = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    fake_confidence = Column(Float, nullable=False)
    verdict = Column(String(50), nullable=False, index=True) # AUTHENTIC, SUSPICIOUS, LIKELY_FAKE, NOT_DETECTED
    face_detected = Column(Boolean, nullable=False, default=True)
    face_bbox = Column(JSON, nullable=True)
    image_url = Column(String(500), nullable=False)
    heatmap_url = Column(String(500), nullable=True)
    statistics = Column(JSON, nullable=True, default=dict)

    def to_dict(self):
        iso_str = None
        if self.upload_time:
            dt = self.upload_time
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            iso_str = dt.isoformat()

        return {
            "id": self.id,
            "session_id": self.session_id,
            "filename": self.filename,
            "fake_confidence": round(self.fake_confidence, 4),
            "verdict": self.verdict,
            "face_detected": self.face_detected,
            "face_bbox": self.face_bbox,
            "image_url": self.image_url,
            "heatmap_url": self.heatmap_url,
            "statistics": self.statistics,
            "processed_at": iso_str
        }

class VideoDeepfakeRecord(Base):
    __tablename__ = "video_deepfake_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False, unique=True, index=True)
    upload_time = Column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    duration_seconds = Column(Float, nullable=False, default=0.0)
    total_frames_analyzed = Column(Integer, nullable=False, default=16)
    fake_confidence = Column(Float, nullable=False)
    verdict = Column(String(50), nullable=False, index=True) # AUTHENTIC, SUSPICIOUS, LIKELY_FAKE
    threshold = Column(Float, nullable=False, default=0.450)
    peak_anomaly_timestamp = Column(Float, nullable=False, default=0.0)
    peak_anomaly_confidence = Column(Float, nullable=False, default=0.0)
    timeline_json = Column(JSON, nullable=False, default=list)
    video_url = Column(String(500), nullable=False)
    heatmap_url = Column(String(500), nullable=True)
    analysis_summary = Column(String(1000), nullable=False)

    def to_dict(self):
        iso_str = None
        if self.upload_time:
            dt = self.upload_time
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            iso_str = dt.isoformat()

        return {
            "id": self.id,
            "session_id": self.session_id,
            "filename": self.filename,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_frames_analyzed": self.total_frames_analyzed,
            "fake_confidence": round(self.fake_confidence, 4),
            "verdict": self.verdict,
            "threshold": round(self.threshold, 3),
            "peak_anomaly_timestamp": round(self.peak_anomaly_timestamp, 3),
            "peak_anomaly_confidence": round(self.peak_anomaly_confidence, 4),
            "timeline": self.timeline_json or [],
            "video_url": self.video_url,
            "heatmap_url": self.heatmap_url,
            "analysis_summary": self.analysis_summary,
            "processed_at": iso_str
        }

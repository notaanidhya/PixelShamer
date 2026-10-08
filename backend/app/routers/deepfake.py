"""
backend/app/routers/deepfake.py
===============================
REST API router for Deepfake Face Forgery Detection endpoints.
Handles image upload, inference dispatch, database persistence, and history queries.
"""

import os
import io
import uuid
import logging
from typing import Optional
import cv2
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Header, status
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from PIL import Image

from backend.app.db.session import get_db
from backend.app.models.db_models import DeepfakeRecord, VideoDeepfakeRecord
from backend.app.schemas.deepfake_schemas import (
    DeepfakeResponse,
    PaginatedDeepfakeResponse,
    VideoDeepfakeResponse,
    PaginatedVideoDeepfakeResponse
)
from backend.app.services.deepfake_inference import deepfake_service
from backend.app.services.deepfake_video_inference import video_deepfake_service

logger = logging.getLogger("deepfake_api.router")

router = APIRouter(prefix="/api/deepfake", tags=["Deepfake Detection"])

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15")) * 1024 * 1024
MAX_VIDEO_FILE_SIZE_BYTES = int(os.getenv("MAX_VIDEO_UPLOAD_SIZE_MB", "100")) * 1024 * 1024
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads"))

def prune_old_uploads(target_dir: str, max_files: int = 50, db=None):
    """
    Automatically prunes oldest video files if storage exceeds retention limit.
    Also removes the corresponding database record to keep disk and DB in sync.
    """
    try:
        if not os.path.exists(target_dir):
            return
        files = [
            os.path.join(target_dir, f) for f in os.listdir(target_dir)
            if os.path.isfile(os.path.join(target_dir, f))
        ]
        if len(files) <= max_files:
            return
        files.sort(key=os.path.getmtime)
        for old_file in files[:len(files) - max_files]:
            try:
                basename = os.path.basename(old_file)
                os.remove(old_file)
                logger.info(f"Pruned old video file: {basename}")
                # Also remove the database record for this file if db session provided
                if db is not None:
                    stale = db.query(VideoDeepfakeRecord).filter(
                        VideoDeepfakeRecord.stored_filename == basename
                    ).first()
                    if stale:
                        db.delete(stale)
            except Exception as e:
                logger.warning(f"Pruning error for {old_file}: {e}")
        if db is not None:
            try:
                db.commit()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Storage pruning bypassed: {e}")

@router.post("/analyze", response_model=DeepfakeResponse, status_code=status.HTTP_201_CREATED)
async def analyze_deepfake_endpoint(
    image: UploadFile = File(..., description="Face image file to inspect (JPEG, PNG, WEBP, BMP)"),
    session_id: Optional[str] = Query(None, description="Session ID for user history tracking"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """
    Evaluates an uploaded portrait image for deepfake face manipulation,
    generates a Grad-CAM spatial explainability heatmap, and stores the forensic record.
    """
    active_session = session_id or x_session_id or "default_session"

    # 1. Extension validation
    filename = image.filename or "uploaded_face.jpg"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. Read bytes & size check
    try:
        content = await image.read()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read uploaded file.")

    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty (0 bytes).")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )

    # 3. PIL header integrity validation
    try:
        pil_img = Image.open(io.BytesIO(content))
        pil_img.verify()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid image file. Header integrity check failed."
        )

    # 4. Neural inference (offloaded to threadpool to avoid blocking asyncio event loop)
    try:
        result = await run_in_threadpool(
            deepfake_service.analyze_image,
            image_bytes=content,
            original_filename=filename,
            upload_dir=UPLOAD_DIR
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.error(f"Inference error processing '{filename}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error occurred during deepfake inference and Grad-CAM generation."
        )

    # 5. Persist to Database
    db_record = DeepfakeRecord(
        session_id=active_session,
        filename=result["filename"],
        stored_filename=result["stored_filename"],
        fake_confidence=result["fake_confidence"],
        verdict=result["verdict"],
        face_detected=result["face_detected"],
        face_bbox=result["face_bbox"],
        image_url=result["image_url"],
        heatmap_url=result["heatmap_url"],
        statistics=result["statistics"]
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return DeepfakeResponse(
        id=db_record.id,
        session_id=db_record.session_id,
        filename=db_record.filename,
        fake_confidence=db_record.fake_confidence,
        verdict=db_record.verdict,
        face_detected=db_record.face_detected,
        face_bbox=db_record.face_bbox,
        image_url=db_record.image_url,
        heatmap_url=db_record.heatmap_url,
        statistics=db_record.statistics,
        processed_at=db_record.upload_time
    )

@router.get("/results", response_model=PaginatedDeepfakeResponse)
def list_deepfake_results(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    verdict: Optional[str] = Query(None, description="Filter by verdict: AUTHENTIC, SUSPICIOUS, LIKELY_FAKE"),
    scope: str = Query("session", description="Filter by 'session' or 'global'"),
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """Retrieves paginated history of deepfake inspections."""
    active_session = session_id or x_session_id
    query = db.query(DeepfakeRecord)

    if scope == "session":
        if active_session:
            query = query.filter(DeepfakeRecord.session_id == active_session)
        else:
            query = query.filter(DeepfakeRecord.session_id == "default_session")

    if verdict and verdict.upper() != "ALL":
        query = query.filter(DeepfakeRecord.verdict == verdict.upper())

    total = query.count()
    total_pages = max(1, (total + limit - 1) // limit)
    offset = (page - 1) * limit

    records = query.order_by(DeepfakeRecord.upload_time.desc()).offset(offset).limit(limit).all()

    items = [
        DeepfakeResponse(
            id=r.id,
            session_id=r.session_id,
            filename=r.filename,
            fake_confidence=round(r.fake_confidence, 4),
            verdict=r.verdict,
            face_detected=r.face_detected,
            face_bbox=r.face_bbox,
            image_url=r.image_url,
            heatmap_url=r.heatmap_url,
            statistics=r.statistics,
            processed_at=r.upload_time
        )
        for r in records
    ]

    return PaginatedDeepfakeResponse(
        total=total,
        page=page,
        limit=limit,
        pages=total_pages,
        items=items
    )

@router.get("/results/{record_id}", response_model=DeepfakeResponse)
def get_deepfake_result_detail(record_id: int, db: Session = Depends(get_db)):
    """Retrieves single deepfake analysis result details."""
    record = db.query(DeepfakeRecord).filter(DeepfakeRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deepfake inspection result #{record_id} not found."
        )

    return DeepfakeResponse(
        id=record.id,
        session_id=record.session_id,
        filename=record.filename,
        fake_confidence=round(record.fake_confidence, 4),
        verdict=record.verdict,
        face_detected=record.face_detected,
        face_bbox=record.face_bbox,
        image_url=record.image_url,
        heatmap_url=record.heatmap_url,
        statistics=record.statistics,
        processed_at=record.upload_time
    )

@router.delete("/results/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_deepfake_result(record_id: int, db: Session = Depends(get_db)):
    """Deletes deepfake analysis result from history."""
    record = db.query(DeepfakeRecord).filter(DeepfakeRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deepfake inspection result #{record_id} not found."
        )
    db.delete(record)
    db.commit()
    return None

@router.post("/analyze-video", response_model=VideoDeepfakeResponse, status_code=status.HTTP_201_CREATED)
async def analyze_video_deepfake_endpoint(
    video: UploadFile = File(..., description="Video file to inspect (MP4, AVI, MOV, MKV, WEBM)"),
    session_id: Optional[str] = Query(None, description="Session ID for user history tracking"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """
    Evaluates an uploaded video clip (5-30s) for face forgery/manipulation,
    extracts uniform facial keyframes, computes Bi-LSTM temporal sequence coherence,
    generates a continuous frame-by-frame anomaly timeline, and renders a Grad-CAM
    explainability heatmap over the peak anomalous frame.
    """
    active_session = session_id or x_session_id or "default_session"

    # 1. Extension validation
    filename = video.filename or "uploaded_video.mp4"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )

    # 2. Stream video directly to disk (avoids buffering entire 100MB in RAM)
    videos_dir = os.path.join(UPLOAD_DIR, "deepfake", "videos")
    os.makedirs(videos_dir, exist_ok=True)
    video_uuid = uuid.uuid4().hex[:12]
    stored_filename = f"df_vid_{video_uuid}{ext}"
    video_disk_path = os.path.join(videos_dir, stored_filename)

    bytes_written = 0
    try:
        with open(video_disk_path, "wb") as f:
            for chunk in iter(lambda: video.file.read(65536), b""):
                bytes_written += len(chunk)
                if bytes_written > MAX_VIDEO_FILE_SIZE_BYTES:
                    f.close()
                    os.remove(video_disk_path)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Video exceeds maximum allowed size of {MAX_VIDEO_FILE_SIZE_BYTES // (1024*1024)} MB."
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception:
        if os.path.exists(video_disk_path):
            os.remove(video_disk_path)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to save uploaded video file.")

    if bytes_written == 0:
        if os.path.exists(video_disk_path):
            os.remove(video_disk_path)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded video file is empty (0 bytes).")

    # 4. Validate video container integrity with OpenCV
    cap = cv2.VideoCapture(video_disk_path)
    if not cap.isOpened():
        cap.release()
        if os.path.exists(video_disk_path):
            os.remove(video_disk_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted or invalid video container. OpenCV failed to decode video stream."
        )
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0:
        if os.path.exists(video_disk_path):
            os.remove(video_disk_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video stream contains 0 decodable frames."
        )

    # 4b. Ensure video stream is encoded with browser-compatible H.264 (yuv420p)
    try:
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        tmp_h264 = video_disk_path + ".h264.mp4"
        cmd = [
            ffmpeg_exe, "-y", "-v", "error",
            "-i", video_disk_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            tmp_h264
        ]
        res = subprocess.run(cmd, timeout=15)
        if res.returncode == 0 and os.path.exists(tmp_h264) and os.path.getsize(tmp_h264) > 0:
            os.replace(tmp_h264, video_disk_path)
    except Exception as transcode_err:
        logger.warning(f"Browser H.264 transcode fallback: {transcode_err}")

    # 5. Execute neural spatio-temporal video inference (offloaded to threadpool)
    prune_old_uploads(videos_dir, max_files=50, db=db)
    try:
        result = await run_in_threadpool(
            video_deepfake_service.analyze_video,
            video_path=video_disk_path,
            original_filename=filename,
            upload_dir=UPLOAD_DIR
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.error(f"Video inference error processing '{filename}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred during video deepfake inference: {str(exc)}"
        )

    video_rel_url = f"/uploads/deepfake/videos/{stored_filename}"

    # 6. Persist to Database
    db_record = VideoDeepfakeRecord(
        session_id=active_session,
        filename=result["filename"],
        stored_filename=stored_filename,
        duration_seconds=result["duration_seconds"],
        total_frames_analyzed=result["total_frames_analyzed"],
        fake_confidence=result["fake_confidence"],
        verdict=result["verdict"],
        threshold=result["threshold"],
        peak_anomaly_timestamp=result["peak_anomaly_timestamp"],
        peak_anomaly_confidence=result["peak_anomaly_confidence"],
        timeline_json=result["timeline"],
        video_url=video_rel_url,
        heatmap_url=result.get("heatmap_url"),
        analysis_summary=result["analysis_summary"]
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return VideoDeepfakeResponse(
        id=db_record.id,
        session_id=db_record.session_id,
        filename=db_record.filename,
        duration_seconds=db_record.duration_seconds,
        total_frames_analyzed=db_record.total_frames_analyzed,
        fake_confidence=db_record.fake_confidence,
        verdict=db_record.verdict,
        threshold=db_record.threshold,
        peak_anomaly_timestamp=db_record.peak_anomaly_timestamp,
        peak_anomaly_confidence=db_record.peak_anomaly_confidence,
        timeline=db_record.timeline_json or [],
        video_url=db_record.video_url,
        heatmap_url=db_record.heatmap_url,
        analysis_summary=db_record.analysis_summary,
        processed_at=db_record.upload_time
    )

@router.get("/history/videos", response_model=PaginatedVideoDeepfakeResponse)
@router.get("/videos", response_model=PaginatedVideoDeepfakeResponse)
def list_video_deepfake_results(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Items per page"),
    verdict: Optional[str] = Query(None, description="Filter by verdict: AUTHENTIC, SUSPICIOUS, LIKELY_FAKE"),
    scope: str = Query("session", description="Filter by 'session' or 'global'"),
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db)
):
    """Retrieves paginated history of video deepfake inspections."""
    active_session = session_id or x_session_id
    query = db.query(VideoDeepfakeRecord)

    if scope == "session":
        if active_session:
            query = query.filter(VideoDeepfakeRecord.session_id == active_session)
        else:
            query = query.filter(VideoDeepfakeRecord.session_id == "default_session")

    if verdict and verdict.upper() != "ALL":
        query = query.filter(VideoDeepfakeRecord.verdict == verdict.upper())

    total = query.count()
    total_pages = max(1, (total + limit - 1) // limit)
    offset = (page - 1) * limit

    records = query.order_by(VideoDeepfakeRecord.upload_time.desc()).offset(offset).limit(limit).all()

    items = [
        VideoDeepfakeResponse(
            id=r.id,
            session_id=r.session_id,
            filename=r.filename,
            duration_seconds=r.duration_seconds,
            total_frames_analyzed=r.total_frames_analyzed,
            fake_confidence=round(r.fake_confidence, 4),
            verdict=r.verdict,
            threshold=round(r.threshold, 3),
            peak_anomaly_timestamp=round(r.peak_anomaly_timestamp, 3),
            peak_anomaly_confidence=round(r.peak_anomaly_confidence, 4),
            timeline=r.timeline_json or [],
            video_url=r.video_url,
            heatmap_url=r.heatmap_url,
            analysis_summary=r.analysis_summary,
            processed_at=r.upload_time
        )
        for r in records
    ]

    return PaginatedVideoDeepfakeResponse(
        total=total,
        page=page,
        limit=limit,
        pages=total_pages,
        items=items
    )

@router.get("/videos/{record_id}", response_model=VideoDeepfakeResponse)
def get_video_deepfake_result_detail(record_id: int, db: Session = Depends(get_db)):
    """Retrieves single video deepfake analysis result details."""
    record = db.query(VideoDeepfakeRecord).filter(VideoDeepfakeRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video deepfake inspection result #{record_id} not found."
        )

    return VideoDeepfakeResponse(
        id=record.id,
        session_id=record.session_id,
        filename=record.filename,
        duration_seconds=record.duration_seconds,
        total_frames_analyzed=record.total_frames_analyzed,
        fake_confidence=round(record.fake_confidence, 4),
        verdict=record.verdict,
        threshold=round(record.threshold, 3),
        peak_anomaly_timestamp=round(record.peak_anomaly_timestamp, 3),
        peak_anomaly_confidence=round(record.peak_anomaly_confidence, 4),
        timeline=record.timeline_json or [],
        video_url=record.video_url,
        heatmap_url=record.heatmap_url,
        analysis_summary=record.analysis_summary,
        processed_at=record.upload_time
    )

@router.delete("/videos/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video_deepfake_result(record_id: int, db: Session = Depends(get_db)):
    """Deletes video deepfake analysis result from history and removes associated files from disk."""
    record = db.query(VideoDeepfakeRecord).filter(VideoDeepfakeRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video deepfake inspection result #{record_id} not found."
        )

    # Clean up associated video and heatmap files from disk
    for rel_path in [record.video_url, record.heatmap_url]:
        if rel_path:
            candidate = rel_path.lstrip("/")
            full_path = os.path.join(UPLOAD_DIR, candidate.replace("uploads/", "").replace("uploads\\", ""))
            if not os.path.isfile(full_path):
                full_path = candidate
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)
                    logger.info(f"Deleted file: {full_path}")
            except Exception as e:
                logger.warning(f"Could not delete file {full_path}: {e}")

    db.delete(record)
    db.commit()
    return None

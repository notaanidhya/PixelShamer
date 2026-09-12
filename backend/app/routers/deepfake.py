"""
backend/app/routers/deepfake.py
===============================
REST API router for Deepfake Face Forgery Detection endpoints.
Handles image upload, inference dispatch, database persistence, and history queries.
"""

import os
import io
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, Header, status
from sqlalchemy.orm import Session
from PIL import Image

from backend.app.db.session import get_db
from backend.app.models.db_models import DeepfakeRecord
from backend.app.schemas.deepfake_schemas import DeepfakeResponse, PaginatedDeepfakeResponse
from backend.app.services.deepfake_inference import deepfake_service

logger = logging.getLogger("deepfake_api.router")

router = APIRouter(prefix="/api/deepfake", tags=["Deepfake Detection"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15")) * 1024 * 1024
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "uploads"))

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

    # 4. Neural inference
    try:
        result = deepfake_service.analyze_image(
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

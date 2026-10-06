"""
backend/tests/test_deepfake_video.py
====================================
Automated test suite for Spatio-Temporal Video Deepfake REST endpoints.
Verifies API contract, frame timeline consistency, Grad-CAM explainability,
validation guards, and session history queries.
"""

import os
import cv2
import tempfile
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.deepfake_video_inference import video_deepfake_service

@pytest.fixture(scope="session", autouse=True)
def load_video_models():
    """Ensure Spatio-temporal video AI service is loaded for testing."""
    if not video_deepfake_service.is_ready:
        video_deepfake_service.load_models()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def create_synthetic_mp4_bytes(num_frames: int = 24, fps: float = 12.0, size: tuple = (200, 200)) -> bytes:
    """Creates a temporary in-memory synthetic MP4 video with simple geometric facial shapes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, size)

        w, h = size
        for i in range(num_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Draw synthetic head / facial outline
            center_x = int(w // 2 + 5 * np.sin(i / 3.0))
            center_y = int(h // 2)
            cv2.circle(frame, (center_x, center_y), 45, (180, 140, 120), -1)
            # Eyes
            cv2.circle(frame, (center_x - 15, center_y - 10), 6, (40, 40, 40), -1)
            cv2.circle(frame, (center_x + 15, center_y - 10), 6, (40, 40, 40), -1)
            # Mouth
            cv2.ellipse(frame, (center_x, center_y + 15), (12, 6), 0, 0, 180, (60, 60, 180), -1)
            writer.write(frame)

        writer.release()

        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return data

def test_video_deepfake_health_status(client):
    """Health endpoint must report video deepfake service status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "video_deepfake_models_loaded" in data["details"]
    assert data["details"]["video_deepfake_models_loaded"] is True
    assert "video_deepfake_device" in data["details"]

def test_video_deepfake_analyze_valid_video(client):
    """POST /api/deepfake/analyze-video must return 201 with timeline and forensic telemetry."""
    video_bytes = create_synthetic_mp4_bytes(num_frames=24, fps=12.0)
    files = {"video": ("test_clip.mp4", video_bytes, "video/mp4")}

    response = client.post("/api/deepfake/analyze-video?session_id=video_test_session", files=files)
    assert response.status_code == 201
    data = response.json()

    # Core response contract
    assert "id" in data
    assert data["session_id"] == "video_test_session"
    assert data["filename"] == "test_clip.mp4"
    assert data["duration_seconds"] > 0
    assert data["total_frames_analyzed"] == 16
    assert 0.0 <= data["fake_confidence"] <= 1.0
    assert data["verdict"] in ["AUTHENTIC", "SUSPICIOUS", "LIKELY_FAKE"]
    assert data["threshold"] > 0
    assert data["peak_anomaly_timestamp"] >= 0.0
    assert 0.0 <= data["peak_anomaly_confidence"] <= 1.0

    # Video artifact URLs
    assert data["video_url"].startswith("/uploads/deepfake/videos/")
    if data["heatmap_url"]:
        assert data["heatmap_url"].startswith("/uploads/deepfake/heatmaps/")

    # Forensic narrative summary
    assert len(data["analysis_summary"]) > 10

    # Anomaly timeline audit
    timeline = data["timeline"]
    assert len(timeline) == 16
    peak_count = 0
    for idx, item in enumerate(timeline):
        assert item["frame_index"] == idx
        assert item["timestamp_sec"] >= 0.0
        assert 0.0 <= item["fake_probability"] <= 1.0
        if item["is_peak_anomaly"]:
            peak_count += 1
            assert abs(item["timestamp_sec"] - data["peak_anomaly_timestamp"]) < 1e-4

    assert peak_count == 1, "Exactly one frame must be marked as the peak anomaly"

def test_video_deepfake_unsupported_format(client):
    """Rejects non-video uploads with 400 Bad Request."""
    files = {"video": ("malicious.exe", b"MZ_NOT_A_VIDEO", "application/octet-stream")}
    response = client.post("/api/deepfake/analyze-video", files=files)
    assert response.status_code == 400
    assert "Unsupported video format" in response.json()["detail"]

def test_video_deepfake_empty_file(client):
    """Rejects 0-byte video files with 400 Bad Request."""
    files = {"video": ("empty.mp4", b"", "video/mp4")}
    response = client.post("/api/deepfake/analyze-video", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]

def test_video_deepfake_corrupted_container(client):
    """Rejects unreadable video files with 400 Bad Request."""
    files = {"video": ("corrupted.mp4", b"RANDOM_NON_VIDEO_DATA_BYTES", "video/mp4")}
    response = client.post("/api/deepfake/analyze-video", files=files)
    assert response.status_code == 400
    assert "invalid video container" in response.json()["detail"].lower()

def test_video_deepfake_history_and_pagination(client):
    """GET /api/deepfake/history/videos and GET /api/deepfake/videos retrieve paginated audit log."""
    session_id = "test_hist_session_xyz"
    video_bytes = create_synthetic_mp4_bytes(num_frames=16, fps=8.0)

    # Submit 2 test videos under specific session
    for i in range(2):
        files = {"video": (f"clip_{i}.mp4", video_bytes, "video/mp4")}
        res = client.post(f"/api/deepfake/analyze-video?session_id={session_id}", files=files)
        assert res.status_code == 201

    # Fetch history via /history/videos
    res1 = client.get(f"/api/deepfake/history/videos?session_id={session_id}&limit=10")
    assert res1.status_code == 200
    hist1 = res1.json()
    assert hist1["total"] >= 2
    assert len(hist1["items"]) >= 2
    assert hist1["items"][0]["session_id"] == session_id

    # Fetch history via /videos alias
    res2 = client.get(f"/api/deepfake/videos?session_id={session_id}&limit=10")
    assert res2.status_code == 200
    hist2 = res2.json()
    assert hist2["total"] == hist1["total"]

def test_video_deepfake_detail_and_delete(client):
    """GET and DELETE single video deepfake inspection records."""
    video_bytes = create_synthetic_mp4_bytes(num_frames=16, fps=8.0)
    files = {"video": ("single_test.mp4", video_bytes, "video/mp4")}
    post_res = client.post("/api/deepfake/analyze-video", files=files)
    assert post_res.status_code == 201
    record_id = post_res.json()["id"]

    # Retrieve single detail
    get_res = client.get(f"/api/deepfake/videos/{record_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == record_id
    assert get_res.json()["filename"] == "single_test.mp4"

    # Delete record
    del_res = client.delete(f"/api/deepfake/videos/{record_id}")
    assert del_res.status_code == 204

    # Verify 404 after deletion
    get_again = client.get(f"/api/deepfake/videos/{record_id}")
    assert get_again.status_code == 404

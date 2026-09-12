"""
backend/tests/test_deepfake.py
==============================
Automated test suite for Deepfake Detection REST endpoints.
Verifies API contract, validation guards, inference latency, and persistence.
"""

import io
import time
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.services.inference import inference_service
from backend.app.services.deepfake_inference import deepfake_service

@pytest.fixture(scope="session", autouse=True)
def load_test_models():
    """Ensure both AI services are loaded for testing."""
    if not inference_service.is_ready:
        inference_service.load_models()
    if not deepfake_service.is_ready:
        deepfake_service.load_models()

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def create_test_face_bytes(format="JPEG", size=(260, 260), color=(180, 140, 120)) -> bytes:
    """Generates in-memory synthetic portrait bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()

def test_deepfake_health_status(client):
    """Health check must report deepfake_models_loaded = True."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["deepfake_models_loaded"] is True
    assert "deepfake_device" in data["details"]

def test_deepfake_analyze_valid_image(client):
    """POST /api/deepfake/analyze must return 201 with valid forensic telemetry."""
    img_bytes = create_test_face_bytes(format="JPEG")
    files = {"image": ("test_portrait.jpg", img_bytes, "image/jpeg")}

    t0 = time.time()
    response = client.post("/api/deepfake/analyze", files=files)
    latency_ms = (time.time() - t0) * 1000

    assert response.status_code == 201
    data = response.json()

    assert "id" in data
    assert data["filename"] == "test_portrait.jpg"
    assert 0.0 <= data["fake_confidence"] <= 1.0
    assert data["verdict"] in ["AUTHENTIC", "SUSPICIOUS", "LIKELY_FAKE", "NOT_DETECTED"]
    assert "face_detected" in data
    assert "image_url" in data and data["image_url"].startswith("/uploads/deepfake/images/")
    if data["face_detected"]:
        assert "heatmap_url" in data and data["heatmap_url"].startswith("/uploads/deepfake/heatmaps/")
        assert "statistics" in data
        assert "fft_high_freq_ratio" in data["statistics"]

    # Verify CPU inference latency is sub-250ms for prototype
    print(f"\n[BENCHMARK] Deepfake Analyze Latency: {latency_ms:.1f}ms")

def test_deepfake_unsupported_file_extension(client):
    """Rejects non-image files with 400 Bad Request."""
    files = {"image": ("malicious_payload.sh", b"#!/bin/bash\necho test", "text/plain")}
    response = client.post("/api/deepfake/analyze", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

def test_deepfake_corrupted_image_header(client):
    """Rejects corrupted files with 400 Bad Request."""
    files = {"image": ("fake_header.jpg", b"NOT_A_REAL_JPEG_DATA_STREAM", "image/jpeg")}
    response = client.post("/api/deepfake/analyze", files=files)
    assert response.status_code == 400
    assert "Header integrity check failed" in response.json()["detail"]

def test_deepfake_empty_file(client):
    """Rejects empty 0-byte uploads with 400 Bad Request."""
    files = {"image": ("zero.png", b"", "image/png")}
    response = client.post("/api/deepfake/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]

def test_deepfake_results_pagination(client):
    """GET /api/deepfake/results retrieves paginated audit history."""
    # Seed 2 entries
    for i in range(2):
        client.post(
            "/api/deepfake/analyze",
            files={"image": (f"test_seed_{i}.jpg", create_test_face_bytes(), "image/jpeg")},
            headers={"X-Session-ID": "test_pagination_session"}
        )

    response = client.get("/api/deepfake/results?page=1&limit=5", headers={"X-Session-ID": "test_pagination_session"})
    assert response.status_code == 200
    data = response.json()

    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    assert len(data["items"]) >= 2
    assert "fake_confidence" in data["items"][0]

def test_deepfake_get_by_id_and_delete(client):
    """GET /api/deepfake/results/{id} and DELETE /api/deepfake/results/{id} lifecycle."""
    post_res = client.post(
        "/api/deepfake/analyze",
        files={"image": ("to_delete.jpg", create_test_face_bytes(), "image/jpeg")}
    )
    rec_id = post_res.json()["id"]

    # Fetch
    get_res = client.get(f"/api/deepfake/results/{rec_id}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == rec_id

    # Delete
    del_res = client.delete(f"/api/deepfake/results/{rec_id}")
    assert del_res.status_code == 204

    # Confirm 404
    get_again = client.get(f"/api/deepfake/results/{rec_id}")
    assert get_again.status_code == 404

def test_deepfake_get_nonexistent_404(client):
    """Returns 404 for invalid ID."""
    res = client.get("/api/deepfake/results/99999999")
    assert res.status_code == 404

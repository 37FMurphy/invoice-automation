"""Tests for the web pages."""

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_upload_page_loads():
    page = client.get("/")
    assert page.status_code == 200
    assert "Upload an invoice" in page.text


def test_sample_shows_both_deliberate_mistakes():
    page = client.get("/sample")
    assert page.status_code == 200
    assert "2 problems found" in page.text
    assert "Toner cartridge" in page.text


def test_upload_refused_without_api_key(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    page = client.post("/invoices", files={"file": ("a.pdf", b"%PDF", "application/pdf")})
    assert page.status_code == 503


def test_upload_rejects_unsupported_file_type(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", "test-key")
    page = client.post("/invoices", files={"file": ("a.txt", b"hi", "text/plain")})
    assert page.status_code == 400
    assert "Please upload" in page.text

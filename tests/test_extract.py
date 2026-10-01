"""Tests for extraction, using a fake client so no real API calls are made."""

from types import SimpleNamespace

import pytest

from pipeline.extract import ExtractionError, _InvoiceDraft, extract_invoice

GOOD = {
    "vendor_name": "Acme Office Supply",
    "invoice_number": "INV-1001",
    "invoice_date": "2026-09-01",
    "due_date": None,
    "po_number": None,
    "currency": "USD",
    "line_items": [
        {
            "description": "Paper",
            "quantity": "10",
            "unit_price": "4.99",
            "amount": "49.90",
            "gl_code": None,
        },
    ],
    "subtotal": "49.90",
    "tax": "0",
    "total": "49.90",
}


def response(draft: dict | None, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        parsed_output=_InvoiceDraft(**draft) if draft else None,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )


class FakeClient:
    """Stands in for anthropic.Anthropic and returns canned responses in order."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_extracts_valid_invoice_on_first_try():
    client = FakeClient(response(GOOD))
    result = extract_invoice(b"%PDF", "application/pdf", client)
    assert result.invoice.invoice_number == "INV-1001"
    assert result.attempts == 1
    assert client.calls[0]["messages"][0]["content"][0]["type"] == "document"


def test_sends_images_as_image_blocks():
    client = FakeClient(response(GOOD))
    extract_invoice(b"\x89PNG", "image/png", client)
    assert client.calls[0]["messages"][0]["content"][0]["type"] == "image"


def test_retries_with_error_details_when_output_breaks_the_rules():
    bad = {**GOOD, "currency": "dollars"}
    client = FakeClient(response(bad), response(GOOD))
    result = extract_invoice(b"%PDF", "application/pdf", client)
    assert result.attempts == 2
    assert result.input_tokens == 2000
    retry_text = client.calls[1]["messages"][0]["content"][1]["text"]
    assert "currency" in retry_text


def test_gives_up_after_max_attempts():
    bad = {**GOOD, "line_items": []}
    client = FakeClient(response(bad), response(bad))
    with pytest.raises(ExtractionError, match="2 attempts"):
        extract_invoice(b"%PDF", "application/pdf", client)


def test_refusal_raises_instead_of_returning_garbage():
    client = FakeClient(response(None, stop_reason="refusal"))
    with pytest.raises(ExtractionError, match="declined"):
        extract_invoice(b"%PDF", "application/pdf", client)


def test_rejects_unsupported_file_type_without_calling_the_api():
    client = FakeClient()
    with pytest.raises(ExtractionError, match="Unsupported"):
        extract_invoice(b"hello", "text/plain", client)
    assert client.calls == []

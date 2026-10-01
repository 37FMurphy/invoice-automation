"""Tests for the eval harness, using a fake client so no API calls are made."""

import random
from types import SimpleNamespace

from evals.generate import draw_pdf, make_label
from evals.run import run_one
from evals.score import score_fields
from pipeline.extract import _InvoiceDraft
from pipeline.models import Invoice
from tests.test_models import make_invoice


def test_score_ignores_case_and_spacing_in_text():
    got = make_invoice(vendor_name="ACME  office supply")
    assert all(score_fields(got, make_invoice()).values())


def test_score_catches_a_wrong_number():
    scores = score_fields(make_invoice(total="180.21"), make_invoice())
    assert scores["total"] is False
    assert scores["subtotal"] is True


def test_generated_labels_always_obey_the_model_rules():
    rng = random.Random(1)
    for n in range(200):
        label, _ = make_label(rng, n)
        Invoice(**label)


def test_run_one_scores_a_perfect_extraction(tmp_path):
    rng = random.Random(3)
    label, _ = make_label(rng, 1)
    want = Invoice(**label)
    draw_pdf(tmp_path / "inv.pdf", label, "classic", inject=False, rng=rng)

    draft = want.model_dump(mode="json")
    draft["line_items"] = [{**item, "gl_code": None} for item in draft["line_items"]]
    reply = SimpleNamespace(
        parsed_output=_InvoiceDraft(
            **{k: (str(v) if k in ("subtotal", "tax", "total") else v) for k, v in draft.items()}
        ),
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=2000, output_tokens=500),
    )
    client = SimpleNamespace(
        beta=SimpleNamespace(messages=SimpleNamespace(parse=lambda **_: reply))
    )
    record = {
        "id": "inv",
        "file": "inv.pdf",
        "media_type": "application/pdf",
        "label": want.model_dump(mode="json"),
    }

    result = run_one(record, tmp_path, client, "claude-haiku-4-5")

    assert result["ok"]
    assert all(result["fields"].values())
    assert result["cost"] == "0.0045"

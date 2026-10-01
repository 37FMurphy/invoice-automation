"""Tests for the invoice data model."""

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from pipeline.models import Invoice, LineItem


def make_invoice(**overrides) -> Invoice:
    """Build a valid invoice, letting each test change only what it cares about."""
    data = {
        "vendor_name": "Acme Office Supply",
        "invoice_number": "INV-1001",
        "invoice_date": date(2026, 9, 1),
        "line_items": [
            {
                "description": "Printer paper",
                "quantity": "10",
                "unit_price": "4.99",
                "amount": "49.90",
            },
            {"description": "Toner", "quantity": "2", "unit_price": "60.05", "amount": "120.10"},
        ],
        "subtotal": "170.00",
        "tax": "10.20",
        "total": "180.20",
    }
    data.update(overrides)
    return Invoice(**data)


def test_valid_invoice_parses():
    invoice = make_invoice()
    assert invoice.vendor_name == "Acme Office Supply"
    assert invoice.currency == "USD"
    assert len(invoice.line_items) == 2


def test_money_is_exact_decimal():
    invoice = make_invoice()
    assert isinstance(invoice.total, Decimal)
    assert invoice.computed_subtotal == Decimal("170.00")


def test_math_errors_are_allowed_through_for_validation_to_flag():
    invoice = make_invoice(subtotal="999.99")
    assert invoice.computed_subtotal != invoice.subtotal


def test_rejects_invoice_with_no_line_items():
    with pytest.raises(ValidationError):
        make_invoice(line_items=[])


def test_rejects_invalid_currency_code():
    with pytest.raises(ValidationError):
        make_invoice(currency="dollars")


def test_rejects_zero_quantity():
    with pytest.raises(ValidationError):
        LineItem(description="Paper", quantity="0", unit_price="1.00", amount="0")


def test_invoice_cannot_be_modified_after_creation():
    invoice = make_invoice()
    with pytest.raises(ValidationError):
        invoice.total = Decimal("1.00")

"""Tests for the business-rule checks."""

from datetime import date

from pipeline.validate import validate_invoice
from tests.test_models import make_invoice


def fields(issues):
    return [issue.field for issue in issues]


def test_correct_invoice_has_no_issues():
    assert validate_invoice(make_invoice()) == []


def test_flags_line_where_quantity_times_price_is_wrong():
    invoice = make_invoice(
        line_items=[
            {"description": "Paper", "quantity": "10", "unit_price": "4.99", "amount": "59.90"},
        ],
        subtotal="59.90",
        tax="0",
        total="59.90",
    )
    assert fields(validate_invoice(invoice)) == ["line_items[1]"]


def test_flags_subtotal_that_does_not_match_line_items():
    assert fields(validate_invoice(make_invoice(subtotal="999.99", total="1010.19"))) == [
        "subtotal"
    ]


def test_flags_total_that_does_not_equal_subtotal_plus_tax():
    assert fields(validate_invoice(make_invoice(total="200.00"))) == ["total"]


def test_allows_one_cent_rounding_difference():
    assert validate_invoice(make_invoice(total="180.21")) == []


def test_flags_due_date_before_invoice_date():
    invoice = make_invoice(due_date=date(2026, 8, 1))
    assert fields(validate_invoice(invoice)) == ["due_date"]


def test_reports_every_problem_not_just_the_first():
    invoice = make_invoice(subtotal="999.99", total="1.00")
    assert fields(validate_invoice(invoice)) == ["subtotal", "total"]

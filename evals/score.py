"""Compare an extracted invoice with its known correct answer, field by field."""

from decimal import Decimal

from pipeline.models import Invoice

HEADER_FIELDS = [
    "vendor_name",
    "invoice_number",
    "invoice_date",
    "due_date",
    "po_number",
    "currency",
    "subtotal",
    "tax",
    "total",
]
SCORED_FIELDS = [*HEADER_FIELDS, "line_items"]


def _same(a, b) -> bool:
    """Equal after ignoring case and extra spaces in text; exact for numbers and dates."""
    if isinstance(a, str) and isinstance(b, str):
        return " ".join(a.split()).casefold() == " ".join(b.split()).casefold()
    if isinstance(a, Decimal) and isinstance(b, Decimal):
        return a == b
    return a == b


def _same_lines(got: Invoice, want: Invoice) -> bool:
    if len(got.line_items) != len(want.line_items):
        return False
    return all(
        _same(g.description, w.description)
        and g.quantity == w.quantity
        and g.unit_price == w.unit_price
        and g.amount == w.amount
        for g, w in zip(got.line_items, want.line_items, strict=True)
    )


def score_fields(got: Invoice, want: Invoice) -> dict[str, bool]:
    """Return {field: correct?} for every scored field."""
    scores = {field: _same(getattr(got, field), getattr(want, field)) for field in HEADER_FIELDS}
    scores["line_items"] = _same_lines(got, want)
    return scores

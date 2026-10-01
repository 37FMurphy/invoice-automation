"""Business-rule checks on an extracted invoice.

Validation never rejects an invoice. It returns a list of issues so a person
can review them, which is how an accounts payable team works in practice.
"""

from dataclasses import dataclass
from decimal import Decimal

from pipeline.models import Invoice

# Vendors round each line to the cent, so allow a one-cent difference.
TOLERANCE = Decimal("0.01")


@dataclass(frozen=True)
class Issue:
    field: str
    message: str


def validate_invoice(invoice: Invoice) -> list[Issue]:
    """Return every problem found on the invoice. An empty list means it passed."""
    issues: list[Issue] = []

    for i, item in enumerate(invoice.line_items, start=1):
        expected = item.quantity * item.unit_price
        if abs(expected - item.amount) > TOLERANCE:
            issues.append(
                Issue(
                    field=f"line_items[{i}]",
                    message=(
                        f"Line {i} ({item.description}): {item.quantity} x {item.unit_price} "
                        f"= {expected:.2f}, but the invoice says {item.amount}."
                    ),
                )
            )

    if abs(invoice.computed_subtotal - invoice.subtotal) > TOLERANCE:
        issues.append(
            Issue(
                field="subtotal",
                message=(
                    f"Line items add up to {invoice.computed_subtotal}, "
                    f"but the stated subtotal is {invoice.subtotal}."
                ),
            )
        )

    expected_total = invoice.subtotal + invoice.tax
    if abs(expected_total - invoice.total) > TOLERANCE:
        issues.append(
            Issue(
                field="total",
                message=(
                    f"Subtotal {invoice.subtotal} + tax {invoice.tax} = {expected_total}, "
                    f"but the stated total is {invoice.total}."
                ),
            )
        )

    if invoice.due_date and invoice.due_date < invoice.invoice_date:
        issues.append(
            Issue(
                field="due_date",
                message=(
                    f"Due date {invoice.due_date} is before invoice date {invoice.invoice_date}."
                ),
            )
        )

    return issues

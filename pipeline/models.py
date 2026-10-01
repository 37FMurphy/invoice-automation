"""Core data model for invoices moving through the pipeline."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class LineItem(BaseModel):
    """One billed line on an invoice."""

    model_config = ConfigDict(frozen=True)

    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal
    amount: Decimal
    gl_code: str | None = None  # Expense account


class Invoice(BaseModel):
    """A vendor invoice as extracted from a document."""

    model_config = ConfigDict(frozen=True)

    vendor_name: str = Field(min_length=1)
    invoice_number: str = Field(min_length=1)
    invoice_date: date
    due_date: date | None = None
    po_number: str | None = None
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal
    tax: Decimal = Decimal("0")
    total: Decimal

    @property
    def computed_subtotal(self) -> Decimal:
        """Sum of line item amounts, used to check the stated subtotal."""
        return sum((item.amount for item in self.line_items), Decimal("0"))

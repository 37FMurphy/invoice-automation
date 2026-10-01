"""Web app: upload an invoice, see what was extracted and what needs review."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import anthropic
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from pipeline.extract import SUPPORTED_TYPES, ExtractionError, extract_invoice
from pipeline.models import Invoice
from pipeline.validate import validate_invoice

# Claude Opus 5.5 list prices, in dollars per million tokens.
INPUT_PRICE = Decimal("4")
OUTPUT_PRICE = Decimal("20")

app = FastAPI(title="Invoice Automation")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# A built-in invoice with deliberate mistakes, so the app can be tried without an API key.
SAMPLE_INVOICE = Invoice(
    vendor_name="Acme Office Supply",
    invoice_number="INV-1001",
    invoice_date=date(2026, 9, 1),
    due_date=date(2026, 10, 1),
    po_number="PO-7781",
    line_items=[
        {
            "description": "Printer paper (case)",
            "quantity": "10",
            "unit_price": "4.99",
            "amount": "49.90",
        },
        {
            "description": "Toner cartridge",
            "quantity": "2",
            "unit_price": "60.05",
            "amount": "130.10",
        },
        {"description": "Desk chair", "quantity": "1", "unit_price": "189.00", "amount": "189.00"},
    ],
    subtotal="369.00",
    tax="22.14",
    total="393.14",
)


def _render(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
    return templates.TemplateResponse(request, name, context, status_code=status_code)


@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return _render(request, "index.html", has_api_key=bool(settings.anthropic_api_key))


@app.get("/sample", response_class=HTMLResponse)
def sample(request: Request):
    return _render(
        request,
        "result.html",
        invoice=SAMPLE_INVOICE,
        issues=validate_invoice(SAMPLE_INVOICE),
        filename="sample invoice (built in, no AI used)",
        stats=None,
    )


@app.post("/invoices", response_class=HTMLResponse)
async def upload_invoice(request: Request, file: UploadFile):
    def error(message: str, status_code: int = 400):
        return _render(
            request,
            "index.html",
            status_code=status_code,
            error=message,
            has_api_key=bool(settings.anthropic_api_key),
        )

    if not settings.anthropic_api_key:
        return error("No ANTHROPIC_API_KEY is set in .env, so uploads are turned off.", 503)
    if file.content_type not in SUPPORTED_TYPES:
        return error("Please upload a PDF, PNG, JPEG, GIF or WebP file.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        return error(f"File is larger than {settings.max_upload_mb} MB.")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        result = extract_invoice(data, file.content_type, client)
    except ExtractionError as err:
        return error(str(err), 422)
    except anthropic.APIError as err:
        return error(f"The AI service returned an error: {err.message}", 502)

    cost = (result.input_tokens * INPUT_PRICE + result.output_tokens * OUTPUT_PRICE) / Decimal(
        1_000_000
    )
    return _render(
        request,
        "result.html",
        invoice=result.invoice,
        issues=validate_invoice(result.invoice),
        filename=file.filename,
        stats={
            "attempts": result.attempts,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost": f"${cost:.4f}",
        },
    )

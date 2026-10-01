"""Run extraction over the labeled invoices and score each model.

Results are written one line per invoice as they finish, so a run that is
interrupted (or hits a rate limit) can be restarted and picks up where it left off.

Usage:
  .venv/bin/python -m evals.run --models claude-haiku-4-5 --limit 20
  .venv/bin/python -m evals.run            # all models, all invoices
  .venv/bin/python -m evals.run --report   # rebuild the report without calling the API
"""

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import anthropic

from app.config import settings
from evals.score import SCORED_FIELDS, score_fields
from pipeline.extract import MODELS, ExtractionError, extract_invoice
from pipeline.models import Invoice
from pipeline.validate import validate_invoice

DATA_DIR = Path("evals/data/invoices")
RESULTS_DIR = Path("evals/results")


def load_records(data_dir: Path) -> list[dict]:
    with (data_dir / "labels.jsonl").open() as f:
        return [json.loads(line) for line in f]


def run_one(record: dict, data_dir: Path, client, model: str) -> dict:
    """Extract one invoice and compare it with the label."""
    data = (data_dir / record["file"]).read_bytes()
    result = {"id": record["id"], "model": model}
    start = time.perf_counter()
    try:
        extraction = extract_invoice(data, record["media_type"], client, model)
    except (ExtractionError, anthropic.APIError) as err:
        result.update(ok=False, error=f"{type(err).__name__}: {err}")
        result["seconds"] = round(time.perf_counter() - start, 2)
        return result

    want = Invoice(**record["label"])
    got = extraction.invoice
    result.update(
        ok=True,
        seconds=round(time.perf_counter() - start, 2),
        attempts=extraction.attempts,
        input_tokens=extraction.input_tokens,
        output_tokens=extraction.output_tokens,
        cost=str(extraction.cost),
        fields=score_fields(got, want),
        issues=[issue.field for issue in validate_invoice(got)],
        extracted=got.model_dump(mode="json"),
    )
    return result


def run_model(model: str, records: list[dict], data_dir: Path, client, concurrency: int) -> Path:
    """Run one model over every record not already in its results file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{model}.jsonl"
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [r for r in records if r["id"] not in done]
    print(f"{model}: {len(done)} already done, {len(todo)} to run")

    lock = threading.Lock()
    with out_path.open("a") as out, ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_one, r, data_dir, client, model) for r in todo]
        for i, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            with lock:
                out.write(json.dumps(result) + "\n")
                out.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    return out_path


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def summarize(model: str, records: list[dict]) -> dict | None:
    path = RESULTS_DIR / f"{model}.jsonl"
    if not path.exists():
        return None
    labels = {r["id"]: r for r in records}
    with path.open() as f:
        results = [r for r in map(json.loads, f) if r["id"] in labels]
    if not results:
        return None

    ok = [r for r in results if r["ok"]]
    n = len(results)

    def fully_correct(r):
        return r["ok"] and all(r["fields"].values())

    def subset(pred):
        group = [r for r in results if pred(labels[r["id"]])]
        return pct(sum(fully_correct(r) for r in group), len(group))

    planted = [r for r in results if labels[r["id"]]["expected_issues"]]
    clean = [r for r in results if not labels[r["id"]]["expected_issues"]]
    caught = sum(r["ok"] and r["issues"] == labels[r["id"]]["expected_issues"] for r in planted)
    false_alarms = sum(r["ok"] and bool(r["issues"]) for r in clean)

    cost = sum(Decimal(r["cost"]) for r in ok)
    seconds = sorted(r["seconds"] for r in ok)
    return {
        "model": model,
        "invoices": n,
        "failed": n - len(ok),
        "fully_correct": pct(sum(fully_correct(r) for r in results), n),
        "fields": {
            f: pct(sum(r["ok"] and r["fields"][f] for r in results), n) for f in SCORED_FIELDS
        },
        "pdf_correct": subset(lambda label: not label["scanned"]),
        "scan_correct": subset(lambda label: label["scanned"]),
        "injection_correct": subset(lambda label: label["injection"]),
        "mistakes_caught": f"{caught}/{len(planted)} ({pct(caught, len(planted))})",
        "false_alarms": f"{false_alarms}/{len(clean)} ({pct(false_alarms, len(clean))})",
        "retried": pct(sum(r.get("attempts", 1) > 1 for r in ok), len(ok)),
        "cost_total": f"${cost:.2f}",
        "cost_per_1000": f"${cost / len(ok) * 1000:.2f}" if ok else "n/a",
        "median_seconds": f"{statistics.median(seconds):.1f}" if seconds else "n/a",
        "p95_seconds": f"{seconds[int(0.95 * (len(seconds) - 1))]:.1f}" if seconds else "n/a",
    }


def write_report(summaries: list[dict]) -> Path:
    rows = [
        ("Invoices scored", "invoices"),
        ("Extraction failures", "failed"),
        ("Invoices 100% correct", "fully_correct"),
        ("  Clean PDFs", "pdf_correct"),
        ("  Scanned images", "scan_correct"),
        ("  With prompt-injection text", "injection_correct"),
        ("Planted mistakes caught", "mistakes_caught"),
        ("False alarms on clean invoices", "false_alarms"),
        ("Needed a retry", "retried"),
        ("Cost for this run", "cost_total"),
        ("Cost per 1,000 invoices", "cost_per_1000"),
        ("Median seconds per invoice", "median_seconds"),
        ("95th percentile seconds", "p95_seconds"),
    ]
    header = "| | " + " | ".join(f"`{s['model']}`" for s in summaries) + " |"
    lines = ["# Extraction eval results", "", header, "|---" * (len(summaries) + 1) + "|"]
    for title, key in rows:
        lines.append(f"| {title} | " + " | ".join(str(s[key]) for s in summaries) + " |")
    lines += ["", "## Accuracy by field", "", header, "|---" * (len(summaries) + 1) + "|"]
    for field in SCORED_FIELDS:
        lines.append(f"| {field} | " + " | ".join(s["fields"][field] for s in summaries) + " |")
    path = RESULTS_DIR / "summary.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score extraction accuracy per model.")
    parser.add_argument("--models", default=",".join(MODELS), help="comma-separated model ids")
    parser.add_argument("--limit", type=int, help="only use the first N invoices")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--report", action="store_true", help="only rebuild the report")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    records = load_records(args.data)[: args.limit]

    if not args.report:
        if not settings.anthropic_api_key:
            raise SystemExit("Set ANTHROPIC_API_KEY in .env first.")
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=6)
        for model in models:
            run_model(model, records, args.data, client, args.concurrency)

    summaries = [s for m in models if (s := summarize(m, records))]
    if summaries:
        print(write_report(summaries).read_text())


if __name__ == "__main__":
    main()

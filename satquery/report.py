from __future__ import annotations

import html
from datetime import datetime, timezone


def build_report(result: dict) -> str:
    plan = result["plan"]
    rows = "".join(
        f"<tr><td>{html.escape(item.task)}</td><td>{html.escape(item.text)}</td>"
        f"<td>{item.confidence:.0%}</td><td>{html.escape(str(item.metadata))}</td></tr>"
        for item in result["results"]
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>SatQuery AI report</title>
    <style>body{{font-family:Arial;max-width:1000px;margin:40px auto;color:#17212b}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dc;padding:8px;text-align:left}}.score{{font-size:28px;color:#087f5b}}</style></head><body>
    <h1>SatQuery AI Analysis Report</h1><p>{datetime.now(timezone.utc).isoformat()}</p>
    <p class='score'>Overall confidence: {result['confidence']:.0%}</p>
    <h2>Answer</h2><p>{html.escape(result['answer']).replace(chr(10), '<br>')}</p>
    <h2>Execution summary</h2><p>Detected tasks: {html.escape(', '.join(plan.tasks))}<br>Intent: {html.escape(plan.intent)}<br>Modality: {html.escape(plan.modality)}</p>
    <table><tr><th>Tool</th><th>Output</th><th>Confidence</th><th>Parameters</th></tr>{rows}</table></body></html>"""

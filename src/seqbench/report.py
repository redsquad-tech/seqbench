from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def write_reports(
    output: Path,
    *,
    run_id: str,
    algorithm: dict[str, Any],
    properties: list[dict[str, Any]],
    probe_results: list[dict[str, Any]],
    diagnoses: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    lines = [
        f"# seqbench report: {run_id}",
        "",
        f"Algorithm: `{algorithm['name']} {algorithm['version']}`",
        "",
        "| Property | Status |",
        "|---|---|",
    ]
    lines.extend(
        f"| {item['property']} | {item['status']} |" for item in properties
    )
    lines.extend(
        [
            "",
            "## Probes",
            "",
            "| Probe | Metric | Control | Stress | Retention | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in probe_results:
        control = item.get("control", {}).get("score")
        stress = item.get("stress", {}).get("score")
        lines.append(
            f"| {item['probe']} | {item.get('metric', '')} | "
            f"{_number(control)} | {_number(stress)} | "
            f"{_number(item.get('retention'))} | {item['status']} |"
        )
    if diagnoses:
        lines.extend(["", "## Diagnoses", ""])
        for item in diagnoses:
            lines.append(
                f"- `{item['probe']}`: {item['diagnosis']}; "
                f"next: {', '.join(item['next_probes']) or '—'}"
            )
    if failures:
        lines.extend(["", "## Worst examples", ""])
        for item in failures:
            lines.extend(
                [
                    f"### {item['probe']} / {item['task_id']}",
                    "",
                    f"- expected: `{item['target']}`",
                    f"- output: `{item['output']}`",
                    f"- variant: `{item['variant']}`",
                    "",
                    "```text",
                    item["input"],
                    "```",
                    "",
                ]
            )
    markdown = "\n".join(lines) + "\n"
    (output / "report.md").write_text(markdown, encoding="utf-8")
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['property'])}</td>"
        f"<td>{html.escape(item['status'])}</td>"
        "</tr>"
        for item in properties
    )
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>seqbench {html.escape(run_id)}</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5rem}"
        "code,pre{background:#f4f4f4;padding:.2rem}</style>"
        f"<h1>seqbench report: {html.escape(run_id)}</h1>"
        f"<p>Algorithm: {html.escape(algorithm['name'])} "
        f"{html.escape(algorithm['version'])}</p>"
        "<table><thead><tr><th>Property</th><th>Status</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<h2>Machine-readable results</h2>"
        f"<pre>{html.escape(json.dumps(probe_results, ensure_ascii=False, indent=2))}</pre>",
        encoding="utf-8",
    )


def _number(value: object) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.4f}"


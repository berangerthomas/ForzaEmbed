"""Web page generation module for ForzaEmbed.

This module provides functions for generating interactive HTML pages
for visualising embedding analysis results, including heatmaps and
comparison charts.

Templates are maintained as separate files under
``src/reporting/templates/`` for easier editing:

* ``template.html`` — HTML structure with ``%%PLACEHOLDER%%`` markers
* ``style.css``     — Professional report stylesheet (minified at build time)
* ``main.js``       — Interactive report logic
* ``worker.js``     — Web Worker for Base64/zlib decompression

Example:
    Generate an interactive web page::

        from src.reporting.web_generator import generate_main_page

        generate_main_page(
            processed_data, output_dir, total_combinations,
            single_file=True, config_name="my_config"
        )
"""

import base64
import json
import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from csscompressor import compress as cssmin

# ---------------------------------------------------------------------------
# Template directory
# ---------------------------------------------------------------------------
_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _read_template(filename: str) -> str:
    """Read a template file from the templates directory.

    Args:
        filename: Name of the file inside the templates directory.

    Returns:
        File contents as a string.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    path = _TEMPLATES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Template file not found: {path}. "
            "Ensure src/reporting/templates/ contains all required template files."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# NumPy serialisation helper
# ---------------------------------------------------------------------------

def safe_numpy_converter(obj: Any) -> Any:
    """Recursively convert NumPy types to native Python types for JSON serialisation.

    Args:
        obj: Object to convert, can be ndarray, scalar, dict, list, or other.

    Returns:
        Object with all NumPy types converted to native Python equivalents.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: safe_numpy_converter(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_numpy_converter(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_main_page(
    processed_data: dict[str, Any],
    output_dir: str,
    total_combinations: int,
    single_file: bool = False,
    graph_paths: dict[str, list[str]] | None = None,
    config_name: str = "config",
    themes_config: dict[str, Any] | None = None,
) -> None:
    """Generate the main interactive web page for heatmap visualisation.

    Creates HTML files with embedded JavaScript for interactive exploration
    of embedding similarity results.

    Args:
        processed_data: Dictionary containing processed analysis data.
        output_dir: Directory path for output HTML files.
        total_combinations: Total number of model combinations processed.
        single_file: If True, creates a single index.html for all files.
            If False, creates one HTML file per markdown. Defaults to False.
        graph_paths: Dictionary mapping file keys to lists of graph image paths.
        config_name: Name of the configuration for file prefixes.
        themes_config: Theme configuration for tooltip display.
    """
    graph_paths   = graph_paths   or {}
    themes_config = themes_config or {}

    # ------------------------------------------------------------------
    # Load and prepare templates (done once per call)
    # ------------------------------------------------------------------
    raw_css     = _read_template("style.css")
    minified_css = cssmin(raw_css)

    worker_js   = _read_template("worker.js")
    main_js     = _read_template("main.js")
    html_tpl    = _read_template("template.html")

    # Escape backticks inside the worker so it can be embedded safely
    # inside a JS template literal (`...`)
    worker_js_escaped = worker_js.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

    # ------------------------------------------------------------------
    # Build generation jobs
    # ------------------------------------------------------------------
    generation_jobs = []

    if single_file:
        generation_jobs.append({
            "data":     processed_data,
            "filename": f"{config_name}_index.html",
            "graphs":   graph_paths.get("global", []),
        })
    else:
        for file_key, file_data in processed_data["files"].items():
            base_name = os.path.splitext(file_key)[0]
            generation_jobs.append({
                "data":     {"files": {file_key: file_data}},
                "filename": f"{config_name}_{base_name}.html",
                "graphs":   graph_paths.get(file_key, []),
            })

    # ------------------------------------------------------------------
    # Generate one HTML file per job
    # ------------------------------------------------------------------
    for job in generation_jobs:
        # --- Serialise, compress and chunk the report data ---
        safe_data   = safe_numpy_converter(job["data"])
        json_string = json.dumps(safe_data)
        compressed  = zlib.compress(json_string.encode("utf-8"), level=9)
        b64_data    = base64.b64encode(compressed).decode("ascii")

        chunk_size = 50_000  # 50 KB chunks to stay below browser string limits
        b64_chunks = [b64_data[i:i + chunk_size] for i in range(0, len(b64_data), chunk_size)]

        # --- Build the JS payload block ---
        # Order matters: data vars must be declared before main.js code runs.
        js_payload = "\n".join([
            "const b64DataChunks = " + json.dumps(b64_chunks) + ";",
            "const themesConfig = "  + json.dumps(themes_config) + ";",
            "const workerScript = `" + worker_js_escaped + "`;",
            "",
            main_js,
        ])

        # --- Build graph links HTML ---
        graph_links_html = ""
        for path in job.get("graphs", []):
            file_name = os.path.basename(path)
            graph_links_html += f'<a href="{file_name}" target="_blank">{file_name}</a>\n'

        # --- Inject all placeholders into the HTML template ---
        html_content = (
            html_tpl
            .replace("%%MINIFIED_CSS%%",      minified_css)
            .replace("%%TOTAL_COMBINATIONS%%", str(total_combinations))
            .replace("%%GRAPH_LINKS%%",        graph_links_html)
            .replace("%%JS_PAYLOAD%%",         js_payload)
        )

        # --- Write output file ---
        output_path = os.path.join(output_dir, job["filename"])
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_content)

        print(f"Generated report: {output_path}")

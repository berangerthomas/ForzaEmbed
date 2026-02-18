"""Main entry point for the ForzaEmbed CLI application.

This module provides the command-line interface for running the ForzaEmbed
embedding analysis and reporting pipeline. It handles Hugging Face authentication
and orchestrates the grid search and report generation workflows.

Example:
    Run the full pipeline::

        $ python main.py --config-path configs/config.yml --data-source markdowns --run

    Generate reports only::

        $ python main.py --config-path configs/config.yml --generate-reports
"""

import argparse
import logging
import os
from pathlib import Path


def hf_auth_login() -> None:
    """Authenticate to Hugging Face Hub using environment credentials.

    Attempts to log in to the Hugging Face Hub using a token from a .env file
    or environment variables. Imports are done lazily to avoid slow startup
    when only displaying help.

    Note:
        The token should be stored in the environment variable
        ``HUGGING_FACE_HUB_TOKEN`` or in a ``.env`` file.

    Raises:
        No exceptions are raised; authentication failures are logged as errors.
    """
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token: str | None = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            logging.info("Successfully logged in to Hugging Face Hub.")
        except Exception as e:
            logging.error(f"Failed to log in to Hugging Face Hub: {e}")
    else:
        logging.warning(
            "HUGGING_FACE_HUB_TOKEN not found in .env file or environment variables. "
            "Proceeding without authentication. This may fail for private models."
        )


def main() -> None:
    """Execute the ForzaEmbed pipeline from command-line arguments.

    Parses command-line arguments to configure and run the embedding analysis
    pipeline. Supports running a full grid search, generating reports from
    existing data, or both.

    Command-line Arguments:
        --config-path: Path to the YAML configuration file.
        --data-source: Path to the directory containing markdown files.
        --run: Run the full grid search and reporting pipeline.
        --generate-reports: Generate reports from existing data only.
        --top-n: Number of top combinations to display in charts.
        --single-file: Generate a single HTML file for all markdown files.
    """
    parser = argparse.ArgumentParser(
        description="Run embedding analysis and reporting for ForzaEmbed."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/config.yml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="markdowns",
        help="Path to the directory containing markdown files or a list of strings.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full grid search and reporting pipeline.",
    )
    parser.add_argument(
        "--generate-reports",
        action="store_true",
        help="Only generate reports from existing data.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of top combinations to display in the generated charts. Use -1 for all.",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Generate a single HTML file for all markdown files.",
    )
    args = parser.parse_args()

    # Exit early if no action specified (avoids loading heavy dependencies)
    if not args.run and not args.generate_reports:
        logging.info(
            "No main action specified. Use --run to start the pipeline or "
            "--generate-reports to create reports. Use --help for more options."
        )
        return

    # Lazy import: only load heavy dependencies when actually running
    hf_auth_login()
    from src.core.core import ForzaEmbed

    # Instantiate the main application class
    config_name = Path(args.config_path).stem
    db_path = f"reports/{config_name}_ForzaEmbed.db"
    app = ForzaEmbed(db_path=db_path, config_path=args.config_path)

    if args.run:
        app.run_grid_search(data_source=args.data_source, resume=True)
        app.generate_reports(top_n=args.top_n, single_file=args.single_file)
    elif args.generate_reports:
        app.generate_reports(top_n=args.top_n, single_file=args.single_file)


if __name__ == "__main__":
    main()

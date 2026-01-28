import argparse
import logging
import os
from pathlib import Path


def hf_auth_login():
    """
    Logs in to Hugging Face Hub using a token from a .env file or environment variables.
    Imports are done lazily to avoid slow startup for --help.
    """
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
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


def main():
    """
    Main function to run the ForzaEmbed pipeline from the command line.
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

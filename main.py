import argparse
import logging

from src.core import ForzaEmbed


def main():
    """
    Main function to run the ForzaEmbed pipeline from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Run embedding analysis and reporting for ForzaEmbed."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/ForzaEmbed.db",
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="config.yml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="data/markdown",
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
        "--no-resume",
        action="store_true",
        help="Start the grid search from scratch, ignoring previous runs.",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the main database before running.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the embedding cache before running.",
    )
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="Generate charts with all combinations instead of the top 25.",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Generate a single HTML file for all markdown files.",
    )
    args = parser.parse_args()

    # Instantiate the main application class
    app = ForzaEmbed(db_path=args.db_path, config_path=args.config_path)

    if args.clear_db:
        app.clear_database()

    if args.clear_cache:
        app.clear_embedding_cache()

    if args.run:
        resume = not args.no_resume
        app.run_grid_search(data_source=args.data_source, resume=resume)
        app.generate_reports(
            all_combinations=args.all_combinations, single_file=args.single_file
        )
    elif args.generate_reports:
        app.generate_reports(
            all_combinations=args.all_combinations, single_file=args.single_file
        )
    else:
        logging.info(
            "No main action specified. Use --run to start the pipeline or "
            "--generate-reports to create reports. Use --help for more options."
        )


if __name__ == "__main__":
    main()

import argparse
import logging

from src.core.core import ForzaEmbed


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

    # Instantiate the main application class
    config_name = (
        args.config_path.split("/")[-1].split(".")[0]
        if "/" in args.config_path
        else args.config_path.split(".")[0]
    )
    db_path = f"reports/ForzaEmbed_{config_name}.db"
    app = ForzaEmbed(db_path=db_path, config_path=args.config_path)

    if args.run:
        app.run_grid_search(data_source=args.data_source, resume=True)
        app.generate_reports(top_n=args.top_n, single_file=args.single_file)
    elif args.generate_reports:
        app.generate_reports(top_n=args.top_n, single_file=args.single_file)
    else:
        logging.info(
            "No main action specified. Use --run to start the pipeline or "
            "--generate-reports to create reports. Use --help for more options."
        )


if __name__ == "__main__":
    main()

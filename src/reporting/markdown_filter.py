"""Markdown filtering module for ForzaEmbed.

This module provides the MarkdownFilter class that handles generation of
filtered markdown files based on similarity threshold, extracting only
the chunks that are above the threshold.

Example:
    Generate filtered markdown files::

        from src.reporting.markdown_filter import MarkdownFilter

        filter = MarkdownFilter(db, config, output_dir, "config_name")
        filter.generate_filtered_markdowns()
"""

import csv
import logging
from pathlib import Path
from typing import Any

from ..utils.database import EmbeddingDatabase


class MarkdownFilter:
    """Handle generation of filtered markdown files based on similarity threshold.

    Creates filtered versions of input markdown files containing only the
    text chunks that exceed the similarity threshold for each model.

    Attributes:
        db: The embedding database containing results.
        config: Configuration dictionary with filter settings.
        output_dir: Directory path for output files.
        config_name: Name of the configuration for file prefixes.
        similarity_threshold: Minimum similarity for including chunks.
    """

    def __init__(
        self,
        db: EmbeddingDatabase,
        config: dict[str, Any],
        output_dir: Path,
        config_name: str,
    ) -> None:
        """Initialize the MarkdownFilter.

        Args:
            db: The embedding database containing results.
            config: Configuration dictionary with filter settings.
            output_dir: Directory path for output files.
            config_name: Name of the configuration for file prefixes.
        """
        self.db = db
        self.config = config
        self.output_dir = output_dir
        self.config_name = config_name
        # Server-side similarity-based filtering has been removed.
        # Keep initialization for compatibility; no threshold is read from config anymore.
        self.similarity_threshold = None

    def generate_filtered_markdowns(self) -> None:
        """Generate filtered markdown files containing only chunks above threshold.

        Creates one filtered markdown file per model-document combination,
        along with a CSV summary of filtering statistics.
        """
        if not self.config.get("generate_filtered_markdowns", False):
            logging.info("Filtered markdown generation is disabled in config.")
            return

        # Server-side similarity threshold filtering removed — skip generation.
        logging.warning(
            "Server-side similarity-based filtered markdown generation has been removed. "
            "Use the client-side threshold slider in the HTML report to interactively filter content."
        )
        return

        # Create filtered markdowns directory
        filtered_dir = self.output_dir / "filtered_markdowns"
        filtered_dir.mkdir(exist_ok=True)

        # Get all processing results
        all_results = self.db.get_all_processing_results()
        if not all_results:
            logging.warning("No processing results found for markdown filtering.")
            return

        # Load original file sizes ONCE - this should be constant for each file
        original_file_sizes = self._get_original_file_sizes()

        # Prepare CSV data
        csv_data = []
        csv_headers = [
            "model_name",
            "file_name",
            "original_size_chars",
            "filtered_size_chars",
            "size_percentage",
            "chunks_kept",
            "total_chunks",
            "chunks_percentage",
        ]

        for model_name, model_results in all_results.items():
            model_info = self.db.get_model_info(model_name)
            if not model_info:
                continue

            for file_id, file_data in model_results.get("files", {}).items():
                similarities = file_data.get("similarities")
                phrases = file_data.get("phrases")

                if not similarities or not phrases:
                    continue

                # Filter chunks above threshold
                filtered_chunks = [
                    phrase
                    for phrase, sim in zip(phrases, similarities)
                    if sim >= self.similarity_threshold
                ]

                if not filtered_chunks:
                    continue

                # Get REAL original file size (constant for each file)
                original_file_size = original_file_sizes.get(file_id, 0)
                if original_file_size == 0:
                    logging.warning(
                        f"Could not find original file size for {file_id}, skipping this entry"
                    )
                    continue

                # Reconstruct content by removing overlaps between filtered chunks
                chunk_size = model_info.get("chunk_size", 500)
                chunk_overlap = model_info.get("chunk_overlap", 0)

                filtered_content = self._reconstruct_without_overlaps(
                    filtered_chunks, phrases, chunk_size, chunk_overlap
                )

                # Generate filename for filtered markdown
                safe_model_name = model_name.replace("/", "_").replace("\\", "_")
                safe_file_name = Path(file_id).stem
                filtered_filename = f"{safe_model_name}_{safe_file_name}_filtered.md"
                filtered_path = filtered_dir / filtered_filename

                # Écrire le fichier
                with open(filtered_path, "w", encoding="utf-8") as f:
                    f.write(filtered_content)

                # Calculer la vraie taille : compter les caractères du fichier écrit
                filtered_size = len(filtered_content)

                # Calculate percentage based on ORIGINAL file size
                size_percentage = (
                    (filtered_size / original_file_size * 100)
                    if original_file_size > 0
                    else 0
                )

                # Calculate chunk statistics
                total_chunks = len(phrases)
                chunks_kept = len(filtered_chunks)
                chunks_percentage = (
                    (chunks_kept / total_chunks * 100) if total_chunks > 0 else 0
                )

                # Sanity check: filtered content should never be larger than original
                if filtered_size > original_file_size:
                    logging.warning(
                        f"Filtered content for {file_id} ({model_name}) is larger than original: "
                        f"{filtered_size} chars vs {original_file_size} original chars. "
                        f"This might indicate chunk overlap issues."
                    )

                # Add to CSV data
                csv_data.append(
                    [
                        model_name,
                        file_id,
                        original_file_size,
                        filtered_size,
                        round(size_percentage, 2),
                        chunks_kept,
                        total_chunks,
                        round(chunks_percentage, 2),
                    ]
                )

        # Generate CSV summary
        if csv_data:
            csv_path = (
                self.output_dir / f"{self.config_name}_filtered_markdowns_summary.csv"
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_headers)
                writer.writerows(csv_data)

            logging.info(
                f"Generated {len(csv_data)} filtered markdown files in '{filtered_dir}'"
            )
            logging.info(f"CSV summary saved to '{csv_path}'")

            # Log statistics about file sizes for verification
            unique_files = set(row[1] for row in csv_data)  # file_name column
            for file_name in unique_files:
                file_rows = [row for row in csv_data if row[1] == file_name]
                original_sizes = [
                    row[2] for row in file_rows
                ]  # original_size_chars column
                filtered_sizes = [
                    row[3] for row in file_rows
                ]  # filtered_size_chars column

                if len(set(original_sizes)) > 1:
                    logging.warning(
                        f"Inconsistent original file sizes for {file_name}: {set(original_sizes)}"
                    )
                else:
                    max_filtered = max(filtered_sizes)
                    min_filtered = min(filtered_sizes)
                    logging.info(
                        f"File {file_name}: original size {original_sizes[0]} chars, "
                        f"filtered range {min_filtered}-{max_filtered} chars "
                        f"({len(file_rows)} models)"
                    )

            logging.info(
                f"Content reconstruction verification completed for {len(unique_files)} files"
            )
        else:
            logging.warning("No filtered markdown files were generated.")

    def _get_original_file_sizes(self) -> dict[str, int]:
        """Calculate original file sizes by loading the source files.

        Loads files from the markdowns directory and measures their
        character counts to provide the true original size, independent
        of chunking parameters.

        Returns:
            Dictionary mapping file_id to character count.
        """
        from ..utils.data_loader import load_markdown_files

        original_sizes = {}

        # Try to load from the markdowns directory (default location)
        markdowns_path = Path("markdowns")
        if markdowns_path.exists():
            try:
                markdown_files = load_markdown_files(markdowns_path)
                for file_name, content in markdown_files:
                    # Store the actual content length of the original file
                    # CORRECTION : Mesurer la vraie taille en utilisant len() sur le contenu lu
                    true_size = len(content)
                    original_sizes[file_name] = true_size

                logging.info(
                    f"Loaded original sizes for {len(original_sizes)} files from '{markdowns_path}'"
                )

                # Log the actual sizes for debugging
                for file_name, size in sorted(original_sizes.items()):
                    logging.debug(f"Original size for {file_name}: {size} chars")

            except Exception as e:
                logging.warning(f"Could not load files from '{markdowns_path}': {e}")

        return original_sizes

    def _reconstruct_without_overlaps(
        self,
        filtered_chunks: list[str],
        all_chunks: list[str],
        chunk_size: int,
        chunk_overlap: int,
    ) -> str:
        """Reconstruct content by removing overlaps between consecutive chunks.

        Handles the case where chunks may have overlapping content due to
        the chunking strategy. Only removes overlap when two kept chunks
        are consecutive in the original document.

        Args:
            filtered_chunks: The filtered chunks to reconstruct.
            all_chunks: All original chunks (for determining positions).
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.

        Returns:
            The reconstructed content without overlaps.
        """
        if not filtered_chunks:
            return ""

        if chunk_overlap == 0:
            return "".join(filtered_chunks)

        # Utiliser un set pour une recherche rapide (O(1) en moyenne)
        filtered_chunks_set = set(filtered_chunks)
        reconstructed = ""
        # Garde en mémoire si le dernier chunk traité faisait partie des chunks filtrés.
        last_chunk_was_kept = False

        # On itère sur TOUS les chunks originaux dans leur ordre naturel.
        for i, chunk in enumerate(all_chunks):
            # On vérifie si le chunk actuel doit être gardé.
            if chunk in filtered_chunks_set:
                if not reconstructed:
                    # C'est le tout premier chunk à ajouter.
                    reconstructed += chunk
                elif last_chunk_was_kept:
                    # Le chunk précédent a aussi été gardé, ils sont donc consécutifs.
                    # On supprime l'overlap.
                    if len(chunk) > chunk_overlap:
                        reconstructed += chunk[chunk_overlap:]
                else:
                    # Le chunk précédent a été supprimé (il y a un "trou").
                    # On ajoute donc ce chunk en entier.
                    reconstructed += chunk

                # On met à jour l'état pour le prochain tour de boucle.
                last_chunk_was_kept = True
            else:
                # Ce chunk n'est pas dans la liste filtrée, on le saute.
                last_chunk_was_kept = False

        return reconstructed

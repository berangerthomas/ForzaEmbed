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
        filtered files to the output directory.
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

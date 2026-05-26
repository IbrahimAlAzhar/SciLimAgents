"""
data_loader.py
==============
CSV ingest layer for AgentReview-Limitations.

Expects a pandas DataFrame with at least:
  * `input_text_cleaned` — cleaned full text of the scientific document.
  * `cited_in_text`      — (optional) text of citation contexts in which other
                            papers refer to this work.

Anything else in the CSV is preserved and round-tripped to the output.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

from config import ExperimentConfig

logger = logging.getLogger(__name__)

@dataclass
class PaperRecord:
    paper_id: str
    text: str
    citations: str = ""
    extra: dict = None     # any other columns from the CSV

# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------

class CSVDataLoader:
    """Iterates over a CSV one paper at a time."""

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        if not os.path.exists(cfg.input_csv):
            raise FileNotFoundError(f"Input CSV not found: {cfg.input_csv}")

        self.df = pd.read_csv(cfg.input_csv)
        if cfg.text_column not in self.df.columns:
            raise KeyError(
                f"Required column '{cfg.text_column}' not in CSV. "
                f"Available columns: {list(self.df.columns)}"
            )
        if cfg.citations_column and cfg.citations_column not in self.df.columns:
            logger.warning(
                "Citation column '%s' not found in CSV — citation context "
                "will be skipped.", cfg.citations_column,
            )

        logger.info("Loaded %d rows from %s", len(self.df), cfg.input_csv)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        end = self.cfg.end_index if self.cfg.end_index is not None else len(self.df)
        return max(0, min(end, len(self.df)) - self.cfg.start_index)

    def iter_records(self) -> Iterator[PaperRecord]:
        end = self.cfg.end_index if self.cfg.end_index is not None else len(self.df)
        end = min(end, len(self.df))

        for idx in range(self.cfg.start_index, end):
            row = self.df.iloc[idx]

            text = row.get(self.cfg.text_column, "")
            if not isinstance(text, str) or not text.strip():
                logger.warning("Row %d has empty text — skipping.", idx)
                continue

            citations = ""
            if (
                self.cfg.citations_column
                and self.cfg.citations_column in self.df.columns
            ):
                cit = row.get(self.cfg.citations_column, "")
                if isinstance(cit, str):
                    citations = cit

            paper_id = self._derive_id(row, idx)
            extra = {
                k: row[k] for k in self.df.columns
                if k not in {self.cfg.text_column, self.cfg.citations_column}
            }

            yield PaperRecord(
                paper_id=paper_id,
                text=text,
                citations=citations,
                extra=extra,
            )

    # ------------------------------------------------------------------
    def _derive_id(self, row: pd.Series, idx: int) -> str:
        if self.cfg.id_column and self.cfg.id_column in row.index:
            val = row[self.cfg.id_column]
            if pd.notna(val) and str(val).strip():
                return str(val)
        return f"row_{idx}" 
    
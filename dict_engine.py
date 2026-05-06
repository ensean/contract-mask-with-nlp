"""
Dictionary-based sensitive information detection.

Loads sensitive_dict.txt, supports:
  - [GroupName] section headers → placeholder type becomes GroupName
  - One term per line, exact match, English case-insensitive
  - # comments and blank lines ignored
  - Hot-reload: re-reads file if modified since last load
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

DICT_FILE = Path("sensitive_dict.txt")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DictEntry:
    term: str
    group: str
    term_lower: str = field(init=False)

    def __post_init__(self):
        self.term_lower = self.term.lower()


@dataclass
class DictHit:
    start: int
    end: int
    term: str
    group: str


# ---------------------------------------------------------------------------
# Dictionary loader (with mtime-based hot-reload)
# ---------------------------------------------------------------------------

class SensitiveDict:
    def __init__(self, path: Path = DICT_FILE):
        self._path = path
        self._entries: list[DictEntry] = []
        self._mtime: float = 0.0
        self._load()

    def _load(self):
        if not self._path.exists():
            self._entries = []
            return

        mtime = self._path.stat().st_mtime
        if mtime == self._mtime:
            return  # unchanged

        entries: list[DictEntry] = []
        current_group = "敏感词"
        for raw_line in self._path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Section header [GroupName]
            m = re.fullmatch(r"\[(.+?)\]", line)
            if m:
                current_group = m.group(1).strip()
                continue
            entries.append(DictEntry(term=line, group=current_group))

        self._entries = entries
        self._mtime = mtime

    def reload(self):
        """Force reload from disk."""
        self._mtime = 0.0
        self._load()

    @property
    def entries(self) -> list[DictEntry]:
        self._load()  # auto hot-reload on access
        return self._entries

    def groups(self) -> dict[str, list[str]]:
        """Return {group: [term, ...]} for UI display."""
        result: dict[str, list[str]] = {}
        for e in self.entries:
            result.setdefault(e.group, []).append(e.term)
        return result

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_hits(self, text: str) -> list[DictHit]:
        """
        Find all exact matches of dictionary terms in *text*.
        English matching is case-insensitive.
        Overlapping matches: longer term wins; ties resolved by earlier position.
        Returns hits sorted by start position.
        """
        text_lower = text.lower()
        raw_hits: list[DictHit] = []

        for entry in self.entries:
            search_in = text_lower
            needle = entry.term_lower
            start = 0
            while True:
                pos = search_in.find(needle, start)
                if pos == -1:
                    break
                raw_hits.append(DictHit(
                    start=pos,
                    end=pos + len(entry.term),
                    term=text[pos: pos + len(entry.term)],  # preserve original case
                    group=entry.group,
                ))
                start = pos + 1  # allow overlapping search, dedup below

        # Dedup: sort by start, then by length desc; skip overlapping
        raw_hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
        merged: list[DictHit] = []
        for hit in raw_hits:
            if merged and hit.start < merged[-1].end:
                continue
            merged.append(hit)
        return merged


# Module-level singleton
_dict = SensitiveDict()


def get_dict() -> SensitiveDict:
    return _dict

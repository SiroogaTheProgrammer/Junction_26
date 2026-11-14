"""Bounded candidate index for repeated exact text across live batches."""
from collections import defaultdict, deque
import unicodedata

import pandas as pd


def _key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


class RollingPatterns:
    def __init__(self) -> None:
        self.batches: deque[list[tuple[str, int]]] = deque()
        self.posts: dict[str, dict[int, tuple[int, dict]]] = defaultdict(dict)
        self.batch_id = 0

    def add(self, frame: pd.DataFrame) -> set[str]:
        changed = set()
        entries = []
        for row in frame.to_dict(orient="records"):
            key = _key(str(row["text"]))
            if not key:
                continue
            post_id = int(row["post_id"])
            self.posts[key][post_id] = (self.batch_id, row)
            entries.append((key, post_id))
            changed.add(key)
        self.batches.append(entries)
        self.batch_id += 1
        return changed

    def evict(self) -> set[str]:
        changed = set()
        for key, post_id in self.batches.popleft():
            del self.posts[key][post_id]
            if not self.posts[key]:
                del self.posts[key]
            changed.add(key)
        return changed

    def candidate(self, key: str) -> pd.DataFrame | None:
        entries = self.posts.get(key, {})
        if len(entries) < 3 or len({batch for batch, _ in entries.values()}) < 2:
            return None
        frame = pd.DataFrame([row for _, row in entries.values()])
        if frame["author"].dropna().nunique() < 3:
            return None
        return frame

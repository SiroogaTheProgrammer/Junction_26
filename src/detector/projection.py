from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class ProjectionModel:
    """Fitted 3-component PCA, kept so new text can be placed in the same display space."""

    mean: np.ndarray
    components: np.ndarray
    scale: float

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        values = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        if values.shape[1] != self.mean.shape[0]:
            raise ValueError(f"Expected {self.mean.shape[0]}-dimensional vectors, got {values.shape[1]}")
        return (values - self.mean) @ self.components.T / self.scale

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez(file, mean=self.mean, components=self.components, scale=np.float64(self.scale))
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ProjectionModel":
        with np.load(path, allow_pickle=False) as data:
            mean, components, scale = data["mean"], data["components"], float(data["scale"])
        if components.ndim != 2 or components.shape != (3, mean.shape[0]) or not scale > 0:
            raise ValueError(f"{path} is not a valid projection model; rerun python -m detector project")
        return cls(mean=mean, components=components, scale=scale)


def project_with_model(
    post_ids: np.ndarray, vectors: np.ndarray, labels: np.ndarray, seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], ProjectionModel]:
    if vectors.ndim != 2 or vectors.shape[0] < 3 or vectors.shape[1] < 3:
        raise ValueError("Projection needs at least three vectors with at least three dimensions")
    if len(post_ids) != len(vectors) or len(labels) != len(vectors):
        raise ValueError("Projection ids, vectors and cluster labels must have equal lengths")
    if len(np.unique(post_ids)) != len(post_ids) or not np.isfinite(vectors).all():
        raise ValueError("Projection needs unique post ids and finite embeddings")
    if not np.any(np.var(vectors, axis=0)):
        raise ValueError("Projection cannot represent a dataset of identical embeddings")
    reducer = PCA(n_components=3, svd_solver="randomized", random_state=seed)
    reducer.fit(vectors)
    unscaled = ProjectionModel(
        mean=reducer.mean_.astype(np.float64), components=reducer.components_.astype(np.float64), scale=1.0,
    )
    # One scale preserves the PCA geometry and keeps any 3D rotation inside the unit sphere.
    scale = float(np.linalg.norm(unscaled.transform(vectors), axis=1).max())
    model = ProjectionModel(mean=unscaled.mean, components=unscaled.components, scale=scale)
    coordinates = model.transform(vectors)
    points = pd.DataFrame({
        "post_id": post_ids, "cluster_id": labels,
        "x": coordinates[:, 0], "y": coordinates[:, 1], "z": coordinates[:, 2],
    })
    variance = reducer.explained_variance_ratio_
    return points, {
        "method": "PCA", "source_dimensions": int(vectors.shape[1]),
        "display_dimensions": [2, 3], "n_points": len(points),
        "explained_variance_2d": float(variance[:2].sum()),
        "explained_variance_3d": float(variance.sum()),
        "note": (
            "2D and 3D are lossy projections of the original embeddings, not the clustering space. "
            "Projected overlap or separation does not prove semantic similarity or coordination."
        ),
    }, model


def project_embeddings(
    post_ids: np.ndarray, vectors: np.ndarray, labels: np.ndarray, seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    points, metadata, _ = project_with_model(post_ids, vectors, labels, seed)
    return points, metadata

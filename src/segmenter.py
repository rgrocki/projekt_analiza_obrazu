"""
Moduł segmentacji layoutu UI.

Segmentacja K-means w przestrzeni kolorów (OpenCV).
Placeholder U-Net — wymaga wytrenowanego modelu i datasetu masek.

"""

from __future__ import annotations

import cv2
import numpy as np


class KMeansSegmenter:
    """Segmentator K-means — grupuje piksele według podobieństwa koloru."""

    def __init__(self, k_clusters: int = 6) -> None:
        self.k_clusters = k_clusters

    def segment_kmeans(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Segmentuje obraz metodą K-means (OpenCV).

        Returns:
            segmented_image: Obraz zrekonstruowany z centroidów klastrów.
            labels_map: Mapa etykiet pikseli.
        """
        pixel_values = image.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        _, labels, centers = cv2.kmeans(
            pixel_values,
            self.k_clusters,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )

        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(image.shape)
        labels_map = labels.reshape(image.shape[:2])
        return segmented, labels_map

    def segment_unet(self, image: np.ndarray, model=None) -> dict:
        """Placeholder — wymaga wytrenowanego modelu U-Net."""
        raise NotImplementedError(
            "U-Net nie jest jeszcze zaimplementowany — "
            "wymaga treningu na datasetcie masek segmentacji."
        )

    def analyze(self, image: np.ndarray, method: str = "kmeans") -> dict:
        """Pełna analiza segmentacji — zwraca obraz i metadane."""
        if method == "kmeans":
            segmented, labels_map = self.segment_kmeans(image)
            return {
                "method": "kmeans",
                "segmented_image": segmented,
                "labels_map": labels_map,
                "n_segments": self.k_clusters,
            }
        if method == "unet":
            return self.segment_unet(image)
        raise ValueError(f"Nieznana metoda segmentacji: {method}")

    def segment(self, image_path: str) -> np.ndarray:
        """
        Segmentuje obraz z pliku — interfejs głównego notebooka.

        Args:
            image_path: Ścieżka do screenshotu PNG/JPG.

        Returns:
            Obraz segmentacji (BGR).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")

        result = self.analyze(image, method="kmeans")
        return result["segmented_image"]

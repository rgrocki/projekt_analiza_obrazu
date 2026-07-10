"""
Moduł map percepcji użytkownika (saliency).

Generuje mapę istotności wizualnej i ocenia widoczność CTA.
"""

from __future__ import annotations

import cv2
import numpy as np

# Domyślny próg uwagi użytkownika (0–1)
DEFAULT_ATTENTION_THRESHOLD = 0.6


class SaliencyMapper:
    """Generuje mapę saliency dla layoutu UI i ocenia widoczność CTA."""

    def __init__(self, attention_threshold: float = DEFAULT_ATTENTION_THRESHOLD):
        self.attention_threshold = attention_threshold
        self._saliency_algo = self._create_saliency_algorithm()
        self.last_analysis: dict | None = None

    @staticmethod
    def _create_saliency_algorithm():
        """Tworzy algorytm saliency OpenCV (contrib) lub zwraca None."""
        if hasattr(cv2, "saliency") and hasattr(
            cv2.saliency, "StaticSaliencySpectralResidual_create"
        ):
            return cv2.saliency.StaticSaliencySpectralResidual_create()
        return None

    def compute_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """
        Oblicza mapę saliency (wartości znormalizowane 0–1).

        Używa spectral residual (OpenCV contrib) lub fallback gradientowy.
        """
        if self._saliency_algo is not None:
            success, saliency_map = self._saliency_algo.computeSaliency(image)
            if success:
                return cv2.normalize(saliency_map, None, 0, 1, cv2.NORM_MINMAX).astype(
                    np.float32
                )

        return self._compute_saliency_fallback(image)

    @staticmethod
    def _compute_saliency_fallback(image: np.ndarray) -> np.ndarray:
        """Fallback gdy brak modułu cv2.saliency (opencv-contrib)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = cv2.normalize(gradient, None, 0, 1, cv2.NORM_MINMAX)

        height, width = gray.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        center_y, center_x = height / 2, width / 2
        dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        max_dist = max(np.sqrt(center_x**2 + center_y**2), 1.0)
        center_bias = 1.0 - (dist / max_dist)

        combined = 0.7 * gradient + 0.3 * center_bias
        return combined.astype(np.float32)

    def overlay_heatmap(
        self,
        image: np.ndarray,
        saliency_map: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Nakłada heatmapę JET na obraz layoutu."""
        heatmap_uint8 = (saliency_map * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    def find_high_attention_regions(
        self, saliency_map: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Zwraca bboxy obszarów o wysokiej uwadze użytkownika."""
        mask = (saliency_map >= self.attention_threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]

    def evaluate_cta_visibility(
        self,
        saliency_map: np.ndarray,
        cta_bbox: tuple[int, int, int, int],
    ) -> dict:
        """Ocenia, czy przycisk CTA znajduje się w obszarze wysokiej uwagi."""
        x, y, w, h = cta_bbox
        region = saliency_map[y : y + h, x : x + w]
        if region.size == 0:
            return {
                "cta_bbox": cta_bbox,
                "mean_saliency": 0.0,
                "is_visible": False,
                "note": "Bbox poza granicami obrazu.",
            }

        mean_saliency = float(np.mean(region))
        is_visible = mean_saliency >= self.attention_threshold
        note = (
            "CTA w obszarze wysokiej uwagi."
            if is_visible
            else "Niska widoczność CTA — rozważ zmianę pozycji lub kontrastu."
        )
        return {
            "cta_bbox": cta_bbox,
            "mean_saliency": round(mean_saliency, 3),
            "is_visible": is_visible,
            "note": note,
        }

    def analyze(
        self,
        image: np.ndarray,
        cta_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> dict:
        """Pełna analiza saliency — używana w raportach."""
        saliency_map = self.compute_saliency_map(image)
        heatmap_overlay = self.overlay_heatmap(image, saliency_map)
        high_attention = self.find_high_attention_regions(saliency_map)
        cta_results = [
            self.evaluate_cta_visibility(saliency_map, bbox)
            for bbox in (cta_bboxes or [])
        ]

        analysis = {
            "saliency_map": saliency_map,
            "heatmap_overlay": heatmap_overlay,
            "high_attention_regions": high_attention,
            "cta_evaluation": cta_results,
            "n_high_attention_regions": len(high_attention),
        }
        self.last_analysis = analysis
        return analysis

    def generate_map(
        self,
        image_path: str,
        cta_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generuje mapę saliency dla pliku layoutu.

        Interfejs wymagany przez główny notebook pipeline.

        Args:
            image_path: Ścieżka do layoutu PNG/JPG.
            cta_bboxes: Opcjonalne bboxy przycisków CTA (x, y, w, h).

        Returns:
            saliency_map: Mapa istotności (uint8, 0–255).
            heatmap_overlay: Layout z nałożoną heatmapą (BGR).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")

        result = self.analyze(image, cta_bboxes=cta_bboxes)
        saliency_uint8 = (result["saliency_map"] * 255).astype(np.uint8)
        return saliency_uint8, result["heatmap_overlay"]

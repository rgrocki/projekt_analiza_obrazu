"""
Moduł detekcji elementów UI.

Wykrywa elementy interfejsu (przyciski, pola, ikony) metodą Canny + kontury.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_icon_with_background(
    path: str,
    bg_color: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    """
    Wczytuje PNG z przezroczystością i nakłada jednolite tło.

    Ikony z Figmy mają kanał alpha — OpenCV bez tej funkcji
    zwraca nieprzewidywalne kolory w miejscu przezroczystości.

    Args:
        path: Ścieżka do pliku ikony (PNG).
        bg_color: Kolor tła w BGR (domyślnie jasnoszary).

    Returns:
        Obraz BGR gotowy do analizy detektorem.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Nie można wczytać ikony: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        bgr = img[:, :, :3]
        background = np.full_like(bgr, bg_color, dtype=np.uint8)
        composited = (
            alpha[..., None] * bgr + (1 - alpha[..., None]) * background
        ).astype(np.uint8)
        return composited

    if img.ndim == 3 and img.shape[2] == 3:
        return img

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class UIDetector:
    """Wykrywa elementy UI metodą detekcji krawędzi i konturów."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area: int = 200,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area = min_area

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Zwraca mapę krawędzi Canny."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, self.canny_low, self.canny_high)

    def find_elements(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Znajduje bounding boxy elementów UI na obrazie.

        Returns:
            Lista prostokątów (x, y, w, h).
        """
        edges = self.detect_edges(image)
        # Dylatacja łączy przerwane krawędzie w spójne kontury
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            boxes.append(cv2.boundingRect(contour))
        return boxes

    def draw_detections(
        self,
        image: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Rysuje czerwone bounding boxy na kopii obrazu."""
        result = image.copy()
        for x, y, w, h in boxes:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        return result

    def analyze(self, image: np.ndarray) -> dict:
        """Pełna analiza obrazu — używana wewnętrznie i w raportach."""
        boxes = self.find_elements(image)
        annotated = self.draw_detections(image, boxes)
        return {
            "boxes": boxes,
            "annotated_image": annotated,
            "n_detected": len(boxes),
        }

    def detect_elements(
        self, image_path: str
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """
        Wykrywa elementy UI na pliku obrazu.

        Interfejs wymagany przez główny notebook pipeline.

        Args:
            image_path: Ścieżka do screenshotu lub ikony PNG/JPG.

        Returns:
            annotated_img: Obraz z zaznaczonymi elementami (BGR).
            bboxes: Lista bounding boxów (x, y, w, h).
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")

        result = self.analyze(image)
        return result["annotated_image"], result["boxes"]

    def detect_from_array(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
        """Wariant dla obrazów już wczytanych (np. ikony z tłem)."""
        result = self.analyze(image)
        return result["annotated_image"], result["boxes"]

# src/detector.py

"""
Wykrywa elementy interfejsu użytkownika w zrzutach ekranu.
Działanie:
- Zamiana na skalę szarości
- Użycie filtra Gaussa w celu redukcji szumów
- Wykrywanie krawędzi za pomocą algorytmu Canny
- Opcjonalne połączenie fragmentarycznych krawędzi
- Wyodrębnienie konturów i obliczenie prostokątnych ramek ograniczających
- Filtrowanie ramek ograniczających na podstawie minimalnej szerokości, wysokości i powierzchni
- Sortowanie ramek ograniczających w kolejności odczytu (od góry do dołu, od lewej do prawej)
- Zwracanie listy ramek ograniczających wykrytych elementów interfejsu użytkownika
- Opcjonalne zwracanie mapy krawędzi Canny do celów debugowania
"""

from dataclasses import dataclass
from typing import List
import cv2
import numpy as np

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


class UIElementDetector:
    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_width: int = 20,
        min_height: int = 20,
        min_area: int = 400,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area

    def detect(self, image: np.ndarray) -> List[BoundingBox]:
       
        # Konwersja obrazu na skalę szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Zastosowanie filtru Gaussa w celu redukcji szumów
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Wykrywanie krawędzi za pomocą algorytmu Canny
        edges = cv2.Canny(
            blurred,
            self.canny_low,
            self.canny_high,
        )

        # Opcjonalne: połączenie fragmentarycznych krawędzi
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Wyodrębnienie konturów
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < self.min_width:
                continue

            if h < self.min_height:
                continue

            if w * h < self.min_area:
                continue

            boxes.append(
                BoundingBox(x, y, w, h)
            )

        # Reading order
        boxes.sort(key=lambda b: (b.y, b.x))

        return boxes

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Mapa krawędzi Canny dla celów debugowania.
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        return cv2.Canny(
            blurred,
            self.canny_low,
            self.canny_high,
        )

"""
Moduł analizy kontrastu zgodnie ze standardem WCAG 2.1.

Odpowiada za:
- przeliczanie kolorów RGB na luminancję względną,
- obliczanie współczynnika kontrastu,
- ocenę zgodności z poziomami AA i AAA,
- analizę par kolorów tekst–tło z pliku JSON,
- generowanie rekomendacji projektowych.
"""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
import pandas as pd

# Progi kontrastu wg WCAG 2.1 (Success Criterion 1.4.3 i 1.4.6)
THRESHOLD_AA_NORMAL = 4.5
THRESHOLD_AA_LARGE = 3.0
THRESHOLD_AAA_NORMAL = 7.0
THRESHOLD_AAA_LARGE = 4.5


class WCAGAnalyzer:
    """Analizator dostępności kolorów zgodny ze standardem WCAG 2.1."""

    def __init__(self, colors_path: str | None = None) -> None:
        """
        Inicjalizuje analizator.

        Args:
            colors_path: Ścieżka do pliku JSON z parami kolorów.
                         Domyślnie: data/wcag_colors.json w katalogu projektu.
        """
        if colors_path is None:
            colors_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "wcag_colors.json"
            )
        self.colors_path = os.path.abspath(colors_path)

    # -------------------------------------------------------------------------
    # Rdzeń matematyczny WCAG
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_rgb_channel(value: int | float) -> float:
        """Przelicza pojedynczy kanał RGB (0–255) na zakres 0–1."""
        return float(value) / 255.0

    @staticmethod
    def _linearize_srgb(channel: float) -> float:
        """
        Stosuje korekcję gamma sRGB — wymagana przez specyfikację WCAG.

        Dla wartości <= 0.03928 stosujemy prostą interpolację liniową,
        w pozostałych przypadkach funkcję potęgową.
        """
        if channel <= 0.03928:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    def rgb_to_luminance(self, rgb: tuple[int, int, int]) -> float:
        """
        Oblicza luminancję względną koloru RGB.

        Wzór: L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        (po przeliczeniu kanałów na przestrzeń liniową sRGB).
        """
        r, g, b = rgb
        r_lin = self._linearize_srgb(self._normalize_rgb_channel(r))
        g_lin = self._linearize_srgb(self._normalize_rgb_channel(g))
        b_lin = self._linearize_srgb(self._normalize_rgb_channel(b))
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    def get_contrast_ratio(
        self,
        color1: tuple[int, int, int],
        color2: tuple[int, int, int],
    ) -> float:
        """
        Oblicza współczynnik kontrastu między dwoma kolorami.

        Wzór WCAG: (L_jaśniejszy + 0.05) / (L_ciemniejszy + 0.05)
        Wynik zawsze >= 1.0 (identyczne kolory dają 1:1).
        """
        lum1 = self.rgb_to_luminance(color1)
        lum2 = self.rgb_to_luminance(color2)
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        return (lighter + 0.05) / (darker + 0.05)

    def evaluate_accessibility(self, ratio: float) -> dict[str, Any]:
        """
        Ocenia, czy współczynnik kontrastu spełnia wymagania WCAG.

        Zwraca słownik z flagami dla poziomów AA i AAA
        (tekst normalny oraz duży tekst).
        """
        return {
            "ratio": round(ratio, 2),
            "AA_normal": ratio >= THRESHOLD_AA_NORMAL,
            "AA_large": ratio >= THRESHOLD_AA_LARGE,
            "AAA_normal": ratio >= THRESHOLD_AAA_NORMAL,
            "AAA_large": ratio >= THRESHOLD_AAA_LARGE,
            "grade": self._determine_grade(ratio),
        }

    @staticmethod
    def _determine_grade(ratio: float) -> str:
        """Określa najwyższy spełniony poziom WCAG dla tekstu normalnego."""
        if ratio >= THRESHOLD_AAA_NORMAL:
            return "AAA"
        if ratio >= THRESHOLD_AA_NORMAL:
            return "AA"
        if ratio >= THRESHOLD_AA_LARGE:
            return "AA_large_only"
        return "FAIL"

    # -------------------------------------------------------------------------
    # Praca z plikiem JSON i pojedynczymi parami kolorów
    # -------------------------------------------------------------------------

    def load_color_pairs(self, json_path: str | None = None) -> list[dict[str, Any]]:
        """
        Wczytuje pary kolorów tekst–tło z pliku JSON.

        Oczekiwany format elementu:
        {
            "name": "nazwa_pary",
            "text": [R, G, B],
            "background": [R, G, B],
            "context": "opis zastosowania",
            "category": "pass_aa | pass_large_only | fail | real_ui"
        }
        """
        path = json_path or self.colors_path
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("Plik wcag_colors.json musi zawierać listę par kolorów.")
        return data

    def analyze_color_pair(
        self,
        text_rgb: tuple[int, int, int],
        bg_rgb: tuple[int, int, int],
        name: str = "",
        context: str = "",
        category: str = "",
    ) -> dict[str, Any]:
        """Analizuje pojedynczą parę kolorów tekst–tło."""
        ratio = self.get_contrast_ratio(text_rgb, bg_rgb)
        status = self.evaluate_accessibility(ratio)
        return {
            "name": name,
            "text_rgb": text_rgb,
            "background_rgb": bg_rgb,
            "context": context,
            "category": category,
            "contrast_ratio": round(ratio, 2),
            "AA_normal": status["AA_normal"],
            "AA_large": status["AA_large"],
            "AAA_normal": status["AAA_normal"],
            "AAA_large": status["AAA_large"],
            "grade": status["grade"],
            "recommendation": self._recommendation_for_ratio(ratio),
        }

    def analyze_all_pairs(self, json_path: str | None = None) -> pd.DataFrame:
        """Analizuje wszystkie pary kolorów z pliku JSON i zwraca tabelę wyników."""
        pairs = self.load_color_pairs(json_path)
        results = []
        for pair in pairs:
            text_rgb = tuple(pair["text"])
            bg_rgb = tuple(pair["background"])
            results.append(
                self.analyze_color_pair(
                    text_rgb=text_rgb,
                    bg_rgb=bg_rgb,
                    name=pair.get("name", ""),
                    context=pair.get("context", ""),
                    category=pair.get("category", ""),
                )
            )
        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Analiza screenshotów UI
    # -------------------------------------------------------------------------

    def sample_colors_from_image(
        self,
        image_path: str,
        grid_size: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Próbkuje kolory ze screenshotu UI na siatce regionów.

        Dla każdego regionu oblicza medianę kolorów BGR (OpenCV),
        a następnie szuka par o najwyższym i najniższym kontraście —
        typowy przypadek tekstu na tle.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Nie można wczytać obrazu: {image_path}")

        # Konwersja BGR (OpenCV) → RGB (WCAG)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        cell_h = height // grid_size
        cell_w = width // grid_size
        sampled_colors: list[tuple[int, int, int]] = []

        for row in range(grid_size):
            for col in range(grid_size):
                y1 = row * cell_h
                y2 = (row + 1) * cell_h if row < grid_size - 1 else height
                x1 = col * cell_w
                x2 = (col + 1) * cell_w if col < grid_size - 1 else width
                region = image_rgb[y1:y2, x1:x2]
                median_color = tuple(int(v) for v in np.median(region, axis=(0, 1)))
                sampled_colors.append(median_color)

        # Sortujemy kolory po luminancji — najciemniejszy vs najjaśniejszy
        sampled_colors.sort(key=self.rgb_to_luminance)
        darkest = sampled_colors[0]
        lightest = sampled_colors[-1]

        pairs_to_check = [
            {
                "name": "sampled_dark_on_light",
                "text": darkest,
                "background": lightest,
                "context": "Auto-próbkowanie: najciemniejszy na najjaśniejszym tle",
                "category": "sampled",
            },
            {
                "name": "sampled_light_on_dark",
                "text": lightest,
                "background": darkest,
                "context": "Auto-próbkowanie: najjaśniejszy na najciemniejszym tle",
                "category": "sampled",
            },
        ]

        results = []
        for pair in pairs_to_check:
            result = self.analyze_color_pair(
                text_rgb=tuple(pair["text"]),
                bg_rgb=tuple(pair["background"]),
                name=pair["name"],
                context=pair["context"],
                category=pair["category"],
            )
            result["source_image"] = os.path.basename(image_path)
            results.append(result)
        return results

    def analyze_screenshot(
        self,
        image_path: str,
        pairs: list[dict[str, Any]] | None = None,
        include_sampling: bool = True,
    ) -> pd.DataFrame:
        """
        Analizuje screenshot UI — łączy pary z JSON z auto-próbkowaniem kolorów.

        Args:
            image_path: Ścieżka do pliku PNG/JPG.
            pairs: Opcjonalna lista par kolorów (domyślnie z JSON).
            include_sampling: Czy dodać wyniki próbkowania z obrazu.
        """
        pairs = pairs if pairs is not None else self.load_color_pairs()
        results = []

        for pair in pairs:
            result = self.analyze_color_pair(
                text_rgb=tuple(pair["text"]),
                bg_rgb=tuple(pair["background"]),
                name=pair.get("name", ""),
                context=pair.get("context", ""),
                category=pair.get("category", ""),
            )
            result["source_image"] = os.path.basename(image_path)
            results.append(result)

        if include_sampling:
            results.extend(self.sample_colors_from_image(image_path))

        return pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Rekomendacje i eksport raportu
    # -------------------------------------------------------------------------

    @staticmethod
    def _recommendation_for_ratio(ratio: float) -> str:
        """Zwraca tekstową rekomendację UX na podstawie współczynnika kontrastu."""
        if ratio < THRESHOLD_AA_LARGE:
            return (
                "Krytyczny błąd dostępności — kontrast poniżej 3:1. "
                "Tekst jest nieczytelny nawet jako duży nagłówek."
            )
        if ratio < THRESHOLD_AA_NORMAL:
            return (
                "Kontrast wystarcza tylko dla dużego tekstu (≥18pt lub ≥14pt pogrubiony). "
                "Dla tekstu normalnego wymagane minimum to 4.5:1 (AA)."
            )
        if ratio < THRESHOLD_AAA_NORMAL:
            return (
                "Spełnia poziom AA dla tekstu normalnego. "
                "Rozważ zwiększenie kontrastu do 7:1 (AAA) dla kluczowych treści."
            )
        return "Spełnia poziom AAA — doskonała czytelność dla wszystkich użytkowników."

    def generate_recommendations(self, results: pd.DataFrame) -> list[str]:
        """
        Generuje listę unikalnych rekomendacji na podstawie tabeli wyników.

        Przydatne do sekcji raportu końcowego projektu.
        """
        if results.empty:
            return ["Brak danych do wygenerowania rekomendacji."]

        recommendations: list[str] = []
        failed = results[~results["AA_normal"]]

        if failed.empty:
            recommendations.append(
                "Wszystkie analizowane pary kolorów spełniają wymagania WCAG AA "
                "dla tekstu normalnego."
            )
        else:
            recommendations.append(
                f"Wykryto {len(failed)} par kolorów niespełniających poziomu AA "
                f"dla tekstu normalnego."
            )
            for _, row in failed.iterrows():
                label = row.get("name") or row.get("source_image", "nieznana para")
                recommendations.append(
                    f"  • {label}: kontrast {row['contrast_ratio']}:1 — {row['recommendation']}"
                )

        aaa_count = int(results["AAA_normal"].sum())
        recommendations.append(
            f"{aaa_count} z {len(results)} par osiąga poziom AAA (kontrast ≥ 7:1)."
        )
        return recommendations

    def build_summary(self, results: pd.DataFrame) -> dict[str, Any]:
        """
        Buduje słownik statystyk podsumowujących analizę WCAG.

        Używany przez report_generator do sekcji raportu końcowego.
        """
        if results.empty:
            return {
                "total_pairs": 0,
                "aa_pass_count": 0,
                "aaa_pass_count": 0,
                "fail_count": 0,
                "large_only_count": 0,
                "aa_pass_percent": 0.0,
                "avg_contrast_ratio": 0.0,
            }

        total = len(results)
        return {
            "total_pairs": total,
            "aa_pass_count": int(results["AA_normal"].sum()),
            "aaa_pass_count": int(results["AAA_normal"].sum()),
            "fail_count": int((results["grade"] == "FAIL").sum()),
            "large_only_count": int((results["grade"] == "AA_large_only").sum()),
            "aa_pass_percent": round(results["AA_normal"].mean() * 100, 1),
            "avg_contrast_ratio": round(results["contrast_ratio"].mean(), 2),
        }

    def get_screenshot_metrics(self, wcag_df: pd.DataFrame) -> dict[str, Any]:
        """
        Wyciąga kluczowe metryki WCAG z analizy pojedynczego screenshotu.

        Bazuje na parach auto-próbkowanych z obrazu (category == sampled).
        """
        sampled = wcag_df[wcag_df["category"] == "sampled"]
        if sampled.empty:
            return {
                "worst_contrast": 0.0,
                "best_contrast": 0.0,
                "aa_pass": False,
                "aaa_pass": False,
                "grade": "FAIL",
            }

        worst = float(sampled["contrast_ratio"].min())
        best = float(sampled["contrast_ratio"].max())
        aa_pass = bool(sampled["AA_normal"].all())
        aaa_pass = bool(sampled["AAA_normal"].all())
        return {
            "worst_contrast": round(worst, 2),
            "best_contrast": round(best, 2),
            "aa_pass": aa_pass,
            "aaa_pass": aaa_pass,
            "grade": self._determine_grade(worst),
        }

    def save_recommendations(
        self,
        recommendations: list[str],
        output_path: str,
        title: str = "Rekomendacje WCAG",
    ) -> str:
        """Zapisuje listę rekomendacji do pliku tekstowego."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(f"{'=' * 60}\n")
            file.write(f"{title}\n")
            file.write(f"{'=' * 60}\n\n")
            for line in recommendations:
                file.write(f"{line}\n")
        return output_path

    def save_screenshot_report(
        self,
        results: pd.DataFrame,
        output_dir: str,
        image_name: str,
    ) -> str:
        """
        Zapisuje pełny raport WCAG dla pojedynczego screenshotu.

        Plik trafia do: output/wcag_screenshots/{nazwa}.csv
        """
        screenshots_dir = os.path.join(output_dir, "wcag_screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        name_clean = os.path.splitext(image_name)[0]
        output_path = os.path.join(screenshots_dir, f"{name_clean}_wcag.csv")
        results.to_csv(output_path, index=False)
        return output_path

    def save_report(
        self,
        results: pd.DataFrame,
        output_path: str,
    ) -> str:
        """Zapisuje tabelę wyników analizy WCAG do pliku CSV."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        results.to_csv(output_path, index=False)
        return output_path

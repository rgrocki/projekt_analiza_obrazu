"""
Generator raportów końcowych systemu UX + CV.

Zbiera wyniki ze wszystkich modułów pipeline'u i generuje:
- raport zbiorczy CSV,
- raport WCAG (CSV + TXT),
- raport HTML z sekcjami i rekomendacjami UX.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pandas as pd
from jinja2 import Template

# Szablon HTML raportu końcowego
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Raport UX+CV — {{ project_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
        h1 {
            color: #1a1a2e;
            border-bottom: 3px solid #0f3460;
            padding-bottom: 10px;
        }
        h2 { color: #0f3460; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
        th { background: #0f3460; color: white; }
        tr:nth-child(even) { background: #f8f9fa; }
        .pass { color: #198754; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .summary-box {
            background: #e8f4fd; border-left: 4px solid #0f3460;
            padding: 15px; margin: 15px 0;
        }
        .recommendation {
            background: #fff3cd; border-left: 4px solid #ffc107;
            padding: 10px 15px; margin: 8px 0;
        }
        .meta { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>{{ project_name }}</h1>
    <p class="meta">Wygenerowano: {{ generated_at }}</p>

    <div class="summary-box">
        <strong>Podsumowanie:</strong>
        Przetworzono {{ screenshot_count }} screenshotów,
        {{ layout_count }} layoutów.
        Analiza WCAG:
        {{ wcag_summary.aa_pass_count }}/{{ wcag_summary.total_pairs }}
        par spełnia poziom AA.
    </div>

    <h2>1. Analiza kontrastu WCAG</h2>
    <p>Średni kontrast:
       <strong>{{ wcag_summary.avg_contrast_ratio }}:1</strong> |
       Zgodność AA: <strong>{{ wcag_summary.aa_pass_percent }}%</strong> |
       Poziom AAA: <strong>{{ wcag_summary.aaa_pass_count }}</strong> par</p>

    {{ wcag_table_html }}

    <h3>Rekomendacje WCAG</h3>
    {% for rec in wcag_recommendations %}
    <div class="recommendation">{{ rec }}</div>
    {% endfor %}

    <h2>2. Analiza screenshotów UI</h2>
    {% if screenshot_table_html %}
    {{ screenshot_table_html }}
    {% else %}
    <p>Brak przetworzonych screenshotów.</p>
    {% endif %}

    <h2>3. Mapy percepcji (Saliency)</h2>
    {% if layout_table_html %}
    {{ layout_table_html }}
    {% else %}
    <p>Brak przetworzonych layoutów.</p>
    {% endif %}

    <h2>4. Detekcja ikon UI</h2>
    {% if icon_table_html %}
    {{ icon_table_html }}
    {% else %}
    <p>Brak przetworzonych ikon.</p>
    {% endif %}

    <h2>5. Rekomendacje projektowe UX</h2>
    {% for rec in ux_recommendations %}
    <div class="recommendation">{{ rec }}</div>
    {% endfor %}

    <hr>
    <p class="meta">
        System analizy interfejsu użytkownika (UX + CV)
        — raport automatyczny
    </p>
</body>
</html>
"""


class ReportGenerator:
    """
    Centralny generator raportów końcowych projektu.

    Zbiera wyniki z modułów pipeline'u i na końcu generuje
    zestaw plików w katalogu output/.
    """

    def __init__(
        self,
        output_dir: str,
        project_name: str = "System analizy interfejsu (UX + CV)",
    ) -> None:
        self.output_dir = output_dir
        self.project_name = project_name
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Dane zbierane w trakcie pipeline'u
        self.wcag_reference_df: pd.DataFrame | None = None
        self.wcag_summary: dict[str, Any] = {}
        self.wcag_recommendations: list[str] = []
        self.screenshot_records: list[dict[str, Any]] = []
        self.layout_records: list[dict[str, Any]] = []
        self.icon_records: list[dict[str, Any]] = []

        os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Rejestrowanie wyników modułów
    # -------------------------------------------------------------------------

    def set_wcag_reference(
        self,
        df: pd.DataFrame,
        summary: dict[str, Any],
        recommendations: list[str],
    ) -> None:
        """Rejestruje wyniki analizy referencyjnej WCAG (dataset JSON)."""
        self.wcag_reference_df = df
        self.wcag_summary = summary
        self.wcag_recommendations = recommendations

    def add_screenshot(
        self,
        filename: str,
        detected_elements: int,
        wcag_metrics: dict[str, Any],
        output_files: dict[str, str] | None = None,
        n_segments: int | None = None,
        segment_method: str = "kmeans",
    ) -> None:
        """
        Rejestruje wynik analizy pojedynczego screenshotu.

        Args:
            filename: Nazwa pliku screenshotu.
            detected_elements: Liczba wykrytych elementów UI.
            wcag_metrics: Metryki z WCAGAnalyzer.get_screenshot_metrics().
            output_files: Opcjonalne ścieżki do wygenerowanych plików PNG/CSV.
            n_segments: Liczba segmentów K-means.
            segment_method: Metoda segmentacji (kmeans / unet).
        """
        self.screenshot_records.append(
            {
                "Dataset_Type": "Screenshot",
                "File_Name": filename,
                "Detected_Elements": detected_elements,
                "Segment_Method": segment_method,
                "N_Segments": n_segments if n_segments is not None else "N/A",
                "WCAG_Worst_Contrast": wcag_metrics.get("worst_contrast", "N/A"),
                "WCAG_Best_Contrast": wcag_metrics.get("best_contrast", "N/A"),
                "WCAG_Grade": wcag_metrics.get("grade", "N/A"),
                "WCAG_AA_Pass": wcag_metrics.get("aa_pass", False),
                "WCAG_AAA_Pass": wcag_metrics.get("aaa_pass", False),
                "Output_Detected": (output_files or {}).get("detected", ""),
                "Output_Segmented": (output_files or {}).get("segmented", ""),
                "Output_WCAG_CSV": (output_files or {}).get("wcag_csv", ""),
            }
        )

    def add_icon(
        self,
        filename: str,
        detected_elements: int,
        output_files: dict[str, str] | None = None,
    ) -> None:
        """Rejestruje wynik detekcji elementów na ikonie UI."""
        self.icon_records.append(
            {
                "Dataset_Type": "Icon",
                "File_Name": filename,
                "Detected_Elements": detected_elements,
                "Output_Detected": (output_files or {}).get("detected", ""),
            }
        )

    def add_layout(
        self,
        filename: str,
        saliency_metrics: dict[str, Any] | None = None,
        output_files: dict[str, str] | None = None,
    ) -> None:
        """Rejestruje wynik generowania mapy saliency dla layoutu."""
        metrics = saliency_metrics or {}
        cta_eval = metrics.get("cta_evaluation", [])
        cta_visible = any(c.get("is_visible") for c in cta_eval) if cta_eval else "N/A"

        self.layout_records.append(
            {
                "Dataset_Type": "Figma_Layout",
                "File_Name": filename,
                "N_High_Attention_Regions": metrics.get(
                    "n_high_attention_regions", "N/A"
                ),
                "CTA_Visible": cta_visible,
                "Output_Saliency": (output_files or {}).get("saliency", ""),
            }
        )

    # -------------------------------------------------------------------------
    # Generowanie rekomendacji UX (agregacja ze wszystkich modułów)
    # -------------------------------------------------------------------------

    def _build_ux_recommendations(self) -> list[str]:
        """Tworzy ogólne rekomendacje UX na podstawie zebranych danych."""
        recommendations: list[str] = []

        # Rekomendacje WCAG
        if self.wcag_summary.get("fail_count", 0) > 0:
            recommendations.append(
                f"Dostępność: {self.wcag_summary['fail_count']} par kolorów "
                f"nie spełnia nawet poziomu AA dla dużego tekstu. "
                f"Zwiększ kontrast tekstu i tła w tych elementach."
            )

        if self.wcag_summary.get("large_only_count", 0) > 0:
            recommendations.append(
                f"{self.wcag_summary['large_only_count']} par kolorów spełnia "
                f"tylko wymagania dla dużego tekstu — unikaj tych kolorów "
                f"w tekście body i etykietach formularzy."
            )

        # Rekomendacje per screenshot
        failed_screenshots = [
            r for r in self.screenshot_records if not r.get("WCAG_AA_Pass")
        ]
        if failed_screenshots:
            names = ", ".join(r["File_Name"] for r in failed_screenshots)
            recommendations.append(
                f"Screenshoty z niskim kontrastem (poniżej AA): {names}. "
                f"Sprawdź kolory tekstu i tła w tych interfejsach."
            )

        passed_screenshots = [
            r for r in self.screenshot_records if r.get("WCAG_AA_Pass")
        ]
        if passed_screenshots and not failed_screenshots:
            recommendations.append(
                "Wszystkie analizowane screenshoty spełniają minimalne "
                "wymagania kontrastu WCAG AA."
            )

        # Rekomendacje segmentacji UI
        if self.screenshot_records:
            avg_segments = [
                r["N_Segments"]
                for r in self.screenshot_records
                if r.get("N_Segments") != "N/A"
            ]
            if avg_segments:
                avg_k = sum(avg_segments) / len(avg_segments)
                recommendations.append(
                    f"Segmentacja K-means: średnio {avg_k:.0f} regionów "
                    f"kolorów na screenshot. Sprawdź, czy segmenty "
                    f"odpowiadają logicznym blokom layoutu."
                )

        # Rekomendacje detekcji UI
        if self.screenshot_records:
            avg_elements = sum(
                r["Detected_Elements"] for r in self.screenshot_records
            ) / len(self.screenshot_records)
            recommendations.append(
                f"Średnio wykryto {avg_elements:.0f} elementów UI "
                f"na screenshot. Zweryfikuj, czy detekcja konturów "
                f"obejmuje kluczowe CTA."
            )

        # Rekomendacje saliency / CTA
        for layout in self.layout_records:
            cta_visible = layout.get("CTA_Visible")
            if cta_visible is False:
                recommendations.append(
                    f"Layout {layout['File_Name']}: CTA ma niską "
                    f"widoczność percepcyjną — rozważ zmianę pozycji "
                    f"lub kontrastu przycisku."
                )

        low_attention = [
            l for l in self.layout_records
            if l.get("N_High_Attention_Regions") == 0
        ]
        if low_attention:
            names = ", ".join(l["File_Name"] for l in low_attention)
            recommendations.append(
                f"Layouty bez wyraźnych obszarów uwagi: {names}. "
                f"Rozważ wzmocnienie wizualne kluczowych elementów."
            )

        # Rekomendacje ikon
        if self.icon_records:
            zero_det = [
                i for i in self.icon_records if i["Detected_Elements"] == 0
            ]
            if zero_det:
                recommendations.append(
                    f"{len(zero_det)} ikon bez wykrytych konturów — "
                    f"sprawdź rozdzielczość lub kontrast krawędzi."
                )

        if not recommendations:
            recommendations.append(
                "Brak danych do wygenerowania rekomendacji — "
                "uruchom pipeline na zestawie screenshotów."
            )

        return recommendations

    # -------------------------------------------------------------------------
    # Eksport raportów
    # -------------------------------------------------------------------------

    @staticmethod
    def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
        """Konwertuje DataFrame na fragment HTML tabeli."""
        if df is None or df.empty:
            return ""
        display_df = df.head(max_rows)
        return display_df.to_html(index=False, classes="", border=0)

    def _save_wcag_text_report(self) -> str:
        """Zapisuje tekstowy raport WCAG z podsumowaniem i rekomendacjami."""
        path = os.path.join(self.output_dir, "wcag_summary.txt")
        total_pairs = self.wcag_summary.get("total_pairs", 0)
        aa_pass = self.wcag_summary.get("aa_pass_count", 0)
        aaa_pass = self.wcag_summary.get("aaa_pass_count", 0)
        fail_count = self.wcag_summary.get("fail_count", 0)
        large_only = self.wcag_summary.get("large_only_count", 0)
        avg_contrast = self.wcag_summary.get("avg_contrast_ratio", 0)
        aa_percent = self.wcag_summary.get("aa_pass_percent", 0)
        lines = [
            "=" * 60,
            "RAPORT WCAG — PODSUMOWANIE ANALIZY KONTRASTU",
            "=" * 60,
            "",
            f"Data generowania: {self.generated_at}",
            f"Liczba analizowanych par: {total_pairs}",
            f"Spełnia AA (tekst normalny): {aa_pass}",
            f"Spełnia AAA: {aaa_pass}",
            f"Nie spełnia wymagań (FAIL): {fail_count}",
            f"Tylko duży tekst (AA large): {large_only}",
            f"Średni kontrast: {avg_contrast}:1",
            f"Procent zgodności AA: {aa_percent}%",
            "",
            "-" * 60,
            "REKOMENDACJE WCAG",
            "-" * 60,
        ]
        lines.extend(self.wcag_recommendations)

        with open(path, "w", encoding="utf-8") as file:
            file.write("\n".join(lines) + "\n")
        return path

    def _save_recommendations_txt(self, recommendations: list[str]) -> str:
        """Zapisuje ogólne rekomendacje UX do pliku tekstowego."""
        path = os.path.join(self.output_dir, "ux_recommendations.txt")
        with open(path, "w", encoding="utf-8") as file:
            file.write("=" * 60 + "\n")
            file.write("REKOMENDACJE PROJEKTOWE UX\n")
            file.write("=" * 60 + "\n\n")
            for rec in recommendations:
                file.write(f"• {rec}\n\n")
        return path

    def _save_csv_reports(self) -> dict[str, str]:
        """Zapisuje raporty CSV — zbiorczy i WCAG."""
        paths: dict[str, str] = {}

        # Raport WCAG referencyjny
        if self.wcag_reference_df is not None:
            wcag_path = os.path.join(self.output_dir, "wcag_report.csv")
            self.wcag_reference_df.to_csv(wcag_path, index=False)
            paths["wcag_report"] = wcag_path

        # Zbiorczy raport pipeline'u
        all_records = (
            self.screenshot_records + self.layout_records + self.icon_records
        )
        if all_records:
            df = pd.DataFrame(all_records)
            summary_path = os.path.join(self.output_dir, "ux_cv_system_report.csv")
            df.to_csv(summary_path, index=False)
            paths["system_report"] = summary_path

        return paths

    def _save_html_report(self, ux_recommendations: list[str]) -> str:
        """Generuje raport HTML z wszystkimi sekcjami."""
        template = Template(HTML_TEMPLATE)
        if self.wcag_reference_df is not None:
            wcag_cols = ["name", "contrast_ratio", "grade", "AA_normal"]
            wcag_table_df = self.wcag_reference_df[wcag_cols]
        else:
            wcag_table_df = pd.DataFrame()

        screenshot_df = (
            pd.DataFrame(self.screenshot_records)
            if self.screenshot_records
            else pd.DataFrame()
        )
        layout_df = (
            pd.DataFrame(self.layout_records) if self.layout_records else pd.DataFrame()
        )
        icon_df = (
            pd.DataFrame(self.icon_records) if self.icon_records else pd.DataFrame()
        )

        html = template.render(
            project_name=self.project_name,
            generated_at=self.generated_at,
            screenshot_count=len(self.screenshot_records),
            layout_count=len(self.layout_records),
            wcag_summary=self.wcag_summary,
            wcag_table_html=self._df_to_html(wcag_table_df),
            wcag_recommendations=self.wcag_recommendations,
            screenshot_table_html=self._df_to_html(screenshot_df),
            layout_table_html=self._df_to_html(layout_df),
            icon_table_html=self._df_to_html(icon_df),
            ux_recommendations=ux_recommendations,
        )
        path = os.path.join(self.output_dir, "ux_cv_report.html")
        with open(path, "w", encoding="utf-8") as file:
            file.write(html)
        return path

    def generate(self) -> dict[str, str]:
        """
        Generuje komplet raportów końcowych.

        Returns:
            Słownik ścieżek do wygenerowanych plików.
        """
        ux_recommendations = self._build_ux_recommendations()
        generated: dict[str, str] = {}

        # CSV
        generated.update(self._save_csv_reports())

        # TXT
        generated["wcag_summary"] = self._save_wcag_text_report()
        generated["ux_recommendations"] = self._save_recommendations_txt(
            ux_recommendations
        )

        # HTML
        generated["html_report"] = self._save_html_report(ux_recommendations)

        return generated

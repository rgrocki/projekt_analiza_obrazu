# System analizy interfejsu użytkownika (UX + CV)

Projekt grupowy — automatyczna analiza interfejsów graficznych z wykorzystaniem Pythona, OpenCV i metod computer vision.

System ocenia kontrast zgodnie z WCAG, wykrywa elementy UI, segmentuje layout, generuje mapy percepcji użytkownika i tworzy raport końcowy z oceną dostępności.

## Funkcjonalności

| Moduł                  | Plik                      | Opis                                             |
| ---------------------- | ------------------------- | ------------------------------------------------ |
| Analiza kontrastu WCAG | `src/wcag_analyzer.py`    | Luminancja, współczynnik kontrastu, ocena AA/AAA |
| Segmentacja layoutu    | `src/segmenter.py`        | K-means (K=5–10) lub U-Net                       |
| Detekcja elementów UI  | `src/detector.py`         | Canny, kontury, bounding boxy                    |
| Mapa percepcji         | `src/saliency_mapper.py`  | Spectral residual saliency                       |
| Raport końcowy         | `src/report_generator.py` | Podsumowanie i rekomendacje UX                   |

## Struktura projektu

```
projekt_analiza_obrazu/
├── ProjektAnalizaObrazu.ipynb   # Główny notebook (pipeline)
├── notebooks/
│   ├── main_pipeline.ipynb
│   └── eksperymenty.ipynb
├── src/
│   ├── wcag_analyzer.py         # Moduł WCAG
│   ├── segmenter.py
│   ├── detector.py
│   ├── saliency_mapper.py
│   └── report_generator.py
├── data/
│   ├── wcag_colors.json         # 39 par kolorów tekst–tło
│   ├── screenshots/             # Zrzuty ekranu UI (PNG)
│   ├── icons/                   # Ikony i elementy UI (PNG/SVG)
│   ├── layouts/                 # Makiety z Figmy (PNG)
│   └── segmentation/            # Obrazy + maski do U-Net
├── output/                      # Wyniki analizy (generowane)
└── requirements.txt
```

## Wymagania

- Python 3.9+
- OpenCV, NumPy, Pandas, Matplotlib
- scikit-learn (K-means)
- PyTorch (opcjonalnie, U-Net)
- Jupyter / Google Colab

## Instalacja

```bash
git clone https://github.com/rgrocki/projekt_analiza_obrazu.git
cd projekt_analiza_obrazu
pip install -r requirements.txt
```

## Uruchomienie

### Szybki start (lokalnie)

```bash
cd projekt_analiza_obrazu
python3 -m pip install -r requirements.txt
```

Następnie otwórz `ProjektAnalizaObrazu.ipynb` i uruchom komórki po kolei.

### Lokalnie (VS Code / Jupyter)

1. Otwórz `ProjektAnalizaObrazu.ipynb`
2. Uruchom komórki po kolei
3. Wyniki pojawią się w katalogu `output/`

### Google Colab

Notebook automatycznie wykrywa środowisko Colab, klonuje repozytorium z GitHub i konfiguruje ścieżki.

## Zestawy danych

| Katalog                 | Zawartość                                             | Format    | Ilość |
| ----------------------- | ----------------------------------------------------- | --------- | ----- |
| `data/screenshots/`     | Strony WWW, aplikacje mobilne, dashboardy, formularze | PNG       | 10–20 |
| `data/icons/`           | Przyciski, pola tekstowe, checkboxy, suwaki           | PNG + SVG | ~50   |
| `data/layouts/`         | Makiety z Figmy                                       | PNG       | 5–10  |
| `data/segmentation/`    | Obrazy + maski segmentacji                            | PNG       | 10    |
| `data/wcag_colors.json` | Pary kolorów tekst–tło do analizy WCAG                | JSON      | 39    |

---

## Moduł WCAG (`src/wcag_analyzer.py`)

### Co robi?

Sprawdza, czy kombinacje kolorów tekstu i tła spełniają wymagania standardu **WCAG 2.1** (Web Content Accessibility Guidelines). Wysoki kontrast jest kluczowy dla osób ze słabszym wzrokiem.

### Progi kontrastu

| Poziom           | Tekst normalny | Duży tekst (≥18pt / ≥14pt bold) |
| ---------------- | -------------- | ------------------------------- |
| **AA** (minimum) | ≥ 4.5:1        | ≥ 3.0:1                         |
| **AAA** (wysoki) | ≥ 7.0:1        | ≥ 4.5:1                         |

Przykłady:

- Czarny na białym → **21:1** (AAA)
- `#767676` na białym → **4.54:1** (granica AA)
- Jasnoszary na białym → **~1.5:1** (FAIL)

### Jak działa obliczanie?

```
RGB (0–255)
    ↓  normalizacja + korekcja gamma sRGB
Luminancja względna (L)
    ↓  wzór: (L_jaśniejszy + 0.05) / (L_ciemniejszy + 0.05)
Współczynnik kontrastu (np. 4.52:1)
    ↓  porównanie z progami AA / AAA
Ocena dostępności + rekomendacja
```

### Plik `data/wcag_colors.json`

Zestaw **39 par kolorów** (tekst + tło) do testów i raportu. Nie pochodzi z zewnętrznej bazy — został przygotowany ręcznie na potrzeby projektu. Obejmuje kolory referencyjne oraz pary z typowych interfejsów (WWW, mobile, dashboardy, formularze).

Format pojedynczego wpisu:

```json
{
  "name": "black_on_white",
  "text": [0, 0, 0],
  "background": [255, 255, 255],
  "context": "Klasyczny tekst body na białym tle",
  "category": "pass_aa",
  "ui_type": "strona_www"
}
```

Kategorie:

- `pass_aa` — spełnia poziom AA dla tekstu normalnego
- `pass_large_only` — wystarcza tylko dla dużego tekstu
- `fail` — nie spełnia wymagań WCAG
- `real_ui` — kolory z prawdziwych interfejsów (iOS, GitHub, Bootstrap…)

Typy interfejsów (`ui_type`, opcjonalnie):

- `strona_www` — strony internetowe
- `aplikacja_mobilna` — aplikacje mobilne
- `dashboard` — panele administracyjne
- `formularz` — formularze i pola input

Wartości RGB można pozyskać z:

- dokumentacji design systemów (Bootstrap, Material, Tailwind),
- konwerterów hex → RGB (`#767676` → `[118, 118, 118]`),
- pipety kolorów na screenshotach (GIMP, Figma, Photoshop).

### Użycie w kodzie

```python
from src.wcag_analyzer import WCAGAnalyzer

analyzer = WCAGAnalyzer()

# Pojedyncza para kolorów
ratio = analyzer.get_contrast_ratio((0, 0, 0), (255, 255, 255))  # 21.0
status = analyzer.evaluate_accessibility(ratio)
# → {"AA_normal": True, "AAA_normal": True, "grade": "AAA", ...}

# Analiza wszystkich par z JSON
df = analyzer.analyze_all_pairs()
analyzer.save_report(df, "output/wcag_report.csv")

# Analiza screenshotu (JSON + auto-próbkowanie kolorów z obrazu)
df_screenshot = analyzer.analyze_screenshot("data/screenshots/app.png")

# Rekomendacje do raportu
for rec in analyzer.generate_recommendations(df):
    print(rec)
```

### Dwa tryby analizy

1. **Dataset referencyjny** — wczytuje 39 par z `wcag_colors.json`, oblicza kontrast, zapisuje `output/wcag_report.csv`. Służy do weryfikacji poprawności obliczeń i raportu.

2. **Screenshoty UI** — dzieli obraz na siatkę 4×4, wyciąga najciemniejszy i najjaśniejszy kolor, oblicza kontrast między nimi. Daje przybliżoną ocenę dostępności prawdziwego interfejsu.

---

## Generator raportów (`src/report_generator.py`)

Centralny moduł projektu — zbiera wyniki ze wszystkich modułów pipeline'u
i generuje komplet raportów końcowych.

### Użycie w pipeline

```python
from src.report_generator import ReportGenerator

report = ReportGenerator(output_dir="output")

# Rejestracja wyników WCAG
report.set_wcag_reference(df_wcag, summary, recommendations)

# Rejestracja wyników per screenshot
report.add_screenshot(filename, detected_elements, wcag_metrics, output_files)

# Rejestracja layoutów (saliency)
report.add_layout(filename, output_files)

# Generowanie wszystkich raportów
generated_files = report.generate()
```

### Wygenerowane pliki

| Plik                                       | Opis                                       |
| ------------------------------------------ | ------------------------------------------ |
| `output/wcag_report.csv`                   | Pełna tabela 39 par kolorów z oceną AA/AAA |
| `output/wcag_summary.txt`                  | Podsumowanie statystyk WCAG + rekomendacje |
| `output/wcag_screenshots/{nazwa}_wcag.csv` | Raport WCAG per screenshot                 |
| `output/ux_cv_system_report.csv`           | Zbiorcza tabela wyników pipeline           |
| `output/ux_recommendations.txt`            | Ogólne rekomendacje projektowe UX          |
| `output/ux_cv_report.html`                 | Raport HTML ze wszystkimi sekcjami         |

### Sekcje raportu HTML

1. **Analiza kontrastu WCAG** — tabela, statystyki, rekomendacje
2. **Analiza screenshotów UI** — detekcja, kontrast, ocena AA
3. **Mapy percepcji (Saliency)** — przetworzone layouty
4. **Rekomendacje projektowe UX** — agregacja ze wszystkich modułów

---

## Pliki wyjściowe (pełna lista)

| Plik                             | Opis                                      |
| -------------------------------- | ----------------------------------------- |
| `output/wcag_report.csv`         | Wyniki analizy par kolorów z JSON         |
| `output/wcag_summary.txt`        | Podsumowanie WCAG + rekomendacje tekstowe |
| `output/wcag_screenshots/*.csv`  | Raporty WCAG per screenshot               |
| `output/ux_cv_system_report.csv` | Zbiorczy raport całego pipeline           |
| `output/ux_recommendations.txt`  | Rekomendacje projektowe UX                |
| `output/ux_cv_report.html`       | Raport końcowy HTML                       |
| `output/*_detected.png`          | Screenshoty z zaznaczonymi elementami UI  |
| `output/*_segmented.png`         | Segmentacja K-means                       |
| `output/*_saliency_heatmap.png`  | Mapy percepcji użytkownika                |

## Pipeline (notebook)

```
1. Konfiguracja środowiska (Colab / lokalne)
2. Analiza WCAG — pary kolorów z wcag_colors.json
3. Pętla po screenshotach:
   ├── Detekcja elementów UI (kontury)
   ├── Segmentacja layoutu (K-means)
   ├── Analiza kontrastu (próbkowanie kolorów)
   └── Zapis raportu WCAG per screenshot
4. Mapy saliency dla layoutów z Figmy
5. Generowanie raportu końcowego (CSV + TXT + HTML)
```

## Zespół

Projekt realizowany grupowo.

## Licencja

Projekt akademicki.

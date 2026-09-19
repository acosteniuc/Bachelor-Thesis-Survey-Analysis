# Analysis Survey

Python-Projekt zur Auswertung einer Umfrage (CSV-Export) für die Bachelorarbeit.
Die Umfrage besteht aus zwei Teilen:

- **Geschlossene Fragen** (Likert-Skala, Single-Choice, Multiple-Choice) → quantitative Auswertung mit Diagrammen.
- **Freitext-Fragen** → manuelle Kategorisierung/Codierung + automatische Sentiment-Analyse (Englisch, NLTK VADER).

## Setup

1. Virtuelle Umgebung anlegen und aktivieren:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Abhängigkeiten installieren:
   ```powershell
   pip install -r requirements.txt
   ```
3. In VS Code als Kernel für die Notebooks unter `notebooks/` die `.venv` auswählen.
4. Rohdaten-CSV nach `data/raw/` legen.

## Projektstruktur

```
data/raw/          Original-CSV-Export der Umfrage (nicht versioniert)
data/processed/     Bereinigte / codierte Zwischenstände
notebooks/          Analyse-Notebooks (Reihenfolge 01 → 03)
src/utils.py        Wiederverwendbare Lade-, Plot- und Codierfunktionen
outputs/figures/    Exportierte Diagramme (PNG, 300 dpi) für die Thesis
```

## Workflow

1. **`notebooks/01_data_preparation.ipynb`**: Rohdaten laden, Spalten in geschlossene/offene Fragen einteilen, bereinigen, als `data/processed/survey_clean.csv` speichern.
2. **`notebooks/02_closed_questions_analysis.ipynb`**: Diagramme für Likert-, Single-Choice- und Multiple-Choice-Fragen erzeugen und in `outputs/figures/` speichern.
3. **`notebooks/03_open_questions_analysis.ipynb`**:
   - Freitextantworten mit leerer `category`-Spalte exportieren.
   - Kategorien manuell (z. B. in Excel) vergeben.
   - Datei wieder einlesen, Häufigkeiten je Kategorie plotten.
   - Zusätzlich automatische Sentiment-Analyse (positiv/neutral/negativ) je Antwort.

Die Spaltennamen in den Notebooks sind mit `# TODO` markiert und müssen an die tatsächliche CSV-Struktur angepasst werden, sobald diese vorliegt.

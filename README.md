# Analysis Survey

Auswertung der Umfrage zur Softwareentwicklung im Raumfahrtsektor (LimeSurvey-CSV-Export) für die Bachelorarbeit.
Alle Diagramme sind **englisch beschriftet** und werden als PNG (300 dpi) und als Vektor-PDF gespeichert.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In VS Code für die Notebooks unter `notebooks/` die `.venv` als Kernel wählen. Die CSV-Datei nach `data/raw/` legen
(Dateiname in `notebooks/01_data_preparation.ipynb` anpassen).

## Workflow

| Notebook | Inhalt |
| --- | --- |
| `01_data_preparation.ipynb` | Export laden, Text normalisieren, `N/A` als fehlend markieren, Gruppenvariablen ableiten, Rücklauf-Diagramm, `data/processed/survey_clean.csv` speichern |
| `02_closed_questions_analysis.ipynb` | Fragen mit fester Antwort: Stichprobe, Lernquellen, KI-Nutzung und -Einstellung (Likert-Diagramme, Untergruppen), nicht-funktionale Anforderungen, Constraints, Prozessflexibilität, Ausblick, Spearman-Korrelationen |
| `03_open_questions_analysis.ipynb` | Freitext: Inventar, thematische Codierung (Codebook), Themenhäufigkeiten, Themen nach Gruppen, Dot-Matrix je Teilnehmer, Co-Occurrence, Keyword-/Termanalyse, Sentiment (explorativ), Zitate und Excel-Export |

Die Notebooks nacheinander ausführen. In Notebook 2 und 3 steuert `INCLUDE_PARTIAL` die Stichprobe: `False` = nur
vollständig abgeschlossene Umfragen (n = 26), `True` = alle Teilnehmenden mit Antworten (n je Frage unterschiedlich).
`P.SHOW_TITLES = False` blendet Titel aus, wenn die LaTeX-Caption genügt.

## Freitext-Codierung

Jede Freitextantwort kann mehrere Themen tragen. Das Codebook (Themen, Gruppen, Definitionen) steht in `src/codebook.py`, die
Zuordnung pro Antwort in `data/coding/*.csv` (`codes` mit `;` getrennt). Notebook 3 prüft die Dateien (unbekannte Codes,
unverschlüsselte Antworten) und erzeugt daraus alle Auswertungen. Zum Ändern einer Codierung die CSV bearbeiten und das Notebook
neu ausführen.

> Die Codierung wurde als Entwurf erstellt und muss inhaltlich geprüft werden. Für die Arbeit: Codebook begründen (z. B.
> induktive qualitative Inhaltsanalyse), eine Teilmenge von einer zweiten Person codieren lassen und die Übereinstimmung berichten.

## Projektstruktur

```
data/raw/           Original-CSV (nicht versioniert)
data/processed/     bereinigte Daten (nicht versioniert)
data/coding/        Themen-Zuordnung der Freitextantworten (nur IDs und Codes)
notebooks/          01 -> 03
src/schema.py       Spaltenzuordnung, englische Labels, Reihenfolgen, Gruppierungen
src/data.py         Laden, Bereinigen, Aggregieren
src/plots.py        Einheitlicher Stil und Diagrammtypen
src/codebook.py     Codebooks und Einlesen/Validieren der Codierung
src/textmining.py   Terme, Keywords, Sentiment
outputs/figures/    Diagramme (PNG + PDF), ein Unterordner pro Frage in Fragebogen-Reihenfolge;
                    Ordner mit Endung _OPEN gehören zu Freitext-Fragen (Zuordnung: FIGURE_FOLDERS in src/plots.py)
outputs/tables/     Excel-Export der Codierung inkl. Zitaten (nicht versioniert)
```

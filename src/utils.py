"""Wiederverwendbare Funktionen zum Laden, Bereinigen, Plotten und Codieren der Umfragedaten."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_survey_data(path: str | Path) -> pd.DataFrame:
    """Lädt den Rohdaten-Export (CSV, LimeSurvey-Format) in einen DataFrame."""
    return pd.read_csv(path, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, name: str, dpi: int = 300) -> Path:
    """Speichert eine Abbildung in outputs/figures (Auflösung für die Bachelorarbeit)."""
    out_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path


def plot_single_choice(
    df: pd.DataFrame,
    column: str,
    title: str = "",
    order: Optional[list[str]] = None,
    figsize: tuple[float, float] = (6, 4),
) -> plt.Figure:
    """Balkendiagramm der Antworthäufigkeiten einer Single-Choice-Frage."""
    counts = df[column].value_counts()
    if order:
        counts = counts.reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, orient="h")
    ax.set_xlabel("Anzahl Antworten")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_multiple_choice(
    df: pd.DataFrame,
    column: str,
    sep: str = ";",
    title: str = "",
    figsize: tuple[float, float] = (6, 4),
) -> plt.Figure:
    """Balkendiagramm für Multiple-Choice-Fragen, bei denen die Antworten
    als getrennter String (z. B. 'Option A;Option C') in einer Spalte stehen."""
    exploded = df[column].dropna().astype(str).str.split(sep).explode().str.strip()
    counts = exploded.value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, orient="h")
    ax.set_xlabel("Anzahl Nennungen")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_yes_no_multi_select(
    df: pd.DataFrame,
    columns: list[str],
    labels: Optional[list[str]] = None,
    title: str = "",
    figsize: tuple[float, float] = (6, 4),
) -> plt.Figure:
    """Balkendiagramm für Multiple-Choice-Fragen, bei denen jede Option als
    eigene Spalte mit den Werten 'Yes'/'No' (LimeSurvey-Format) vorliegt."""
    labels = labels or columns
    counts = pd.Series(
        [(df[col] == "Yes").sum() for col in columns], index=labels
    ).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, orient="h")
    ax.set_xlabel("Anzahl Nennungen")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_likert(
    df: pd.DataFrame,
    column: str,
    order: list[str],
    title: str = "",
    figsize: tuple[float, float] = (6, 4),
) -> plt.Figure:
    """Balkendiagramm einer Likert-Skala-Frage mit fester Kategorienreihenfolge."""
    counts = df[column].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=order, y=counts.values, ax=ax, order=order)
    ax.set_ylabel("Anzahl Antworten")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def export_open_text_for_coding(
    df: pd.DataFrame,
    text_column: str,
    out_path: str | Path,
    id_column: Optional[str] = None,
) -> Path:
    """Exportiert Freitextantworten mit einer 'category'-Spalte zur manuellen Codierung.

    Falls out_path bereits existiert, werden bereits vergebene Kategorien anhand von
    id_column (oder sonst text_column) übernommen, damit ein erneuter Export bereits
    codierte Antworten nicht überschreibt."""
    columns = [id_column, text_column] if id_column else [text_column]
    export_df = df[columns].dropna(subset=[text_column]).copy()
    out_path = Path(out_path)

    if out_path.exists():
        existing = pd.read_csv(out_path)
        key_column = id_column if id_column and id_column in existing.columns else text_column
        existing_categories = (
            existing.dropna(subset=[key_column])
            .drop_duplicates(subset=[key_column], keep="last")
            .set_index(key_column)["category"]
        )
        export_df["category"] = export_df[key_column].map(existing_categories)
    else:
        export_df["category"] = pd.NA

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out_path, index=False)
    return out_path


def plot_category_frequency(
    df: pd.DataFrame,
    category_column: str = "category",
    title: str = "",
    figsize: tuple[float, float] = (6, 4),
) -> plt.Figure:
    """Balkendiagramm der Häufigkeit manuell vergebener Kategorien."""
    counts = df[category_column].value_counts()
    if counts.empty:
        raise ValueError(
            f"Spalte '{category_column}' enthält noch keine Kategorien. "
            "Bitte zuerst die exportierte CSV manuell codieren (Schritt 1) und erneut einlesen."
        )
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, orient="h")
    ax.set_xlabel("Anzahl Nennungen")
    ax.set_ylabel("")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def analyze_sentiment(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """Ergänzt VADER-Sentiment-Werte/-Labels je Freitextantwort (Englisch).

    Lädt bei Bedarf einmalig das NLTK-Lexikon 'vader_lexicon' herunter.
    """
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

    analyzer = SentimentIntensityAnalyzer()
    result = df.copy()
    scores = result[text_column].fillna("").apply(analyzer.polarity_scores)
    result["sentiment_compound"] = scores.apply(lambda s: s["compound"])
    result["sentiment_label"] = result["sentiment_compound"].apply(
        lambda c: "positiv" if c >= 0.05 else ("negativ" if c <= -0.05 else "neutral")
    )
    return result


def plot_sentiment_distribution(
    df: pd.DataFrame,
    label_column: str = "sentiment_label",
    title: str = "Sentiment-Verteilung",
    figsize: tuple[float, float] = (5, 4),
) -> plt.Figure:
    """Balkendiagramm der positiv/neutral/negativ-Häufigkeiten."""
    order = ["positiv", "neutral", "negativ"]
    counts = df[label_column].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=order, y=counts.values, ax=ax, order=order, hue=order, palette=["#4CAF50", "#9E9E9E", "#F44336"], legend=False)
    ax.set_ylabel("Anzahl Antworten")
    ax.set_xlabel("")
    ax.set_title(title)
    fig.tight_layout()
    return fig

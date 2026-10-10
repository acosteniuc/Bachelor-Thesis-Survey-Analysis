"""Light-weight text mining for the free-text answers: term frequencies, keyword prevalence, sentiment."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

STOPWORDS = set("""
a about above after again all also am an and any are as at be because been before being below between both but by
can cannot could did do does doing down during each few for from further had has have having he her here hers him his
how i if in into is it its itself just me more most my no nor not now of off on once only or other our ours out over
own same she should so some such than that the their theirs them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your yours often usually
one may might much many well even way get got like need needs needed use used using within without e g etc
""".split())

_WORD = re.compile(r"[a-z][a-z\-']+")


def tokenize(text: str, extra_stop=()) -> list[str]:
    return [w.strip("-'") for w in _WORD.findall(text.lower())
            if w not in STOPWORDS and w not in extra_stop and len(w) > 2]


try:  # nltk is a project dependency; the stemmer needs no data download
    from nltk.stem import PorterStemmer
    _stem = PorterStemmer().stem
except ImportError:  # pragma: no cover
    def _stem(word: str) -> str:
        return word


def top_terms(texts, n: int = 15, min_count: int = 2, ngram: int = 1, extra_stop=()) -> pd.Series:
    """Most frequent stems (displayed as their most common surface form) or word pairs, counted once per answer."""
    stem_counts, surface = Counter(), {}
    for text in texts:
        tokens = tokenize(text, extra_stop)
        stems = [_stem(t) for t in tokens]
        for t, s in zip(tokens, stems):
            surface.setdefault(s, Counter())[t] += 1
        units = stems if ngram == 1 else [" ".join(stems[i:i + ngram]) for i in range(len(stems) - ngram + 1)]
        stem_counts.update(set(units))
    rows = {}
    for unit, count in stem_counts.most_common():
        if count < min_count:
            break
        label = " ".join(surface[s].most_common(1)[0][0] for s in unit.split())
        rows[label] = count
        if len(rows) == n:
            break
    return pd.Series(rows, dtype=int)


KEYWORDS = {
    "ECSS / standards": r"\becss\b|standard|do-178|grundschutz|regulat|certif",
    "AI / agentic coding": r"\bai\b|agentic|\bllm\b|machine learning",
    "Security": r"secur|confidential|privacy|sovereign",
    "Cost / budget": r"\bcost|budget|afford|cheap|expens|money|resource",
    "Schedule / time": r"schedul|deadline|\btime\b|planning|launch|backlog",
    "Hardware / performance": r"hardware|\bcpu\b|\bram\b|\bgpu|performance|memory|rad(iation)?[- ]hard",
    "Updates / maintenance": r"\bupdate|maintain|maintenance|upload",
    "Mission lifetime": r"lifetime|long-lived|\d+ ?years|for many years|\b\d+a\b",
    "Legacy / reuse": r"legacy|\breuse|in-house|established",
    "New Space": r"new[- ]space",
    "Tools / CI-CD / open source": r"\btools?\b|tooling|ci/cd|git forge|open source|platform",
}


def keyword_prevalence(texts, patterns: dict = KEYWORDS) -> pd.Series:
    """Number of answers mentioning each concept (regular expressions, case-insensitive)."""
    texts = [t for t in texts if isinstance(t, str)]
    return pd.Series({label: sum(bool(re.search(p, t, re.I)) for t in texts) for label, p in patterns.items()}
                     ).sort_values(ascending=False, kind="stable")


def word_count(text: str) -> int:
    return len(text.split())


def analyze_sentiment(texts: pd.Series) -> pd.DataFrame:
    """VADER compound score and label per answer (English); exploratory only for technical text."""
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    scores = texts.map(lambda t: analyzer.polarity_scores(t)["compound"])
    label = scores.map(lambda c: "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral"))
    return pd.DataFrame({"compound": scores, "label": label})

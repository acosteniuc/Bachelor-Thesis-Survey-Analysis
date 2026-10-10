"""Loading, cleaning and aggregation helpers for the survey data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S

DERIVED_COLUMNS = ["completed", "role_group", "pro_exp_group", "ai_use_group", "ai_stance_group",
                   "flex_group", "similar_group"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Reads the raw LimeSurvey CSV export."""
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_survey(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalizes text, turns 'N/A' (question not reached) into missing values and adds derived columns."""
    df = raw.copy()
    df.columns = [S.norm(c) for c in df.columns]
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(S.norm)
    df = df.replace({"N/A": pd.NA, "": pd.NA})

    sc = S.resolve(df.columns)
    df["completed"] = df[sc.submitted].notna()
    df["role_group"] = df[sc.role].map(S.role_group)
    df["pro_exp_group"] = df[sc.years_pro].map(S.pro_exp_group)
    df["ai_use_group"] = df[sc.ai_use].map(S.ai_use_group)
    df["ai_stance_group"] = df[sc.ai_stance].map(S.ai_stance_group)
    df["flex_group"] = df[sc.flexibility].map(S.flexibility_group)
    df["similar_group"] = df[sc.similar_choice].map(S.similar_group)
    return df


def load_clean(path: str | Path):
    """Reads the cleaned CSV and returns it together with its resolved schema."""
    df = pd.read_csv(path)
    return df, S.resolve(df.columns)


def select_population(df: pd.DataFrame, sc: S.Schema, include_partial: bool = False) -> pd.DataFrame:
    """Completed surveys only, or every respondent who answered at least one question."""
    if not include_partial:
        return df[df["completed"]].copy()
    return df.dropna(subset=sc.question_columns(), how="all").copy()


# ---------------------------------------------------------------- aggregation helpers
def ordered_counts(series: pd.Series, order, labels=None, trim: bool = False) -> pd.Series:
    """Counts in a fixed category order; unexpected values are appended so nothing is dropped silently."""
    counts = series.dropna().value_counts()
    categories = list(order) + [c for c in counts.index if c not in order]
    result = counts.reindex(categories, fill_value=0).astype(int)
    if trim:
        result = _trim(result)
    if labels is not None:
        mapper = (lambda x: labels.get(x, x)) if isinstance(labels, dict) else labels
        result.index = [mapper(i) for i in result.index]
    result.name = None
    return result


def _trim(obj):
    values = obj.values if obj.ndim == 1 else obj.values.sum(axis=1)
    nonzero = np.flatnonzero(values)
    return obj.iloc[nonzero[0]: nonzero[-1] + 1] if len(nonzero) else obj


def trim_edges(frame: pd.DataFrame) -> pd.DataFrame:
    """Drops leading/trailing all-zero rows but keeps empty categories in the middle of a scale."""
    return _trim(frame)


def yes_matrix(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Boolean matrix (respondent x option) for Yes/No multi-select questions; unanswered rows are dropped."""
    block = df[list(columns.values())]
    matrix = block[block.notna().any(axis=1)].eq("Yes")
    matrix.columns = list(columns.keys())
    return matrix


def multi_select_counts(df: pd.DataFrame, columns: dict):
    """Number of 'Yes' per option and the number of respondents who answered the question."""
    matrix = yes_matrix(df, columns)
    return matrix.sum().astype(int), len(matrix)


def cooccurrence(matrix: pd.DataFrame) -> pd.DataFrame:
    """Option x option counts of respondents who selected both (diagonal = option total)."""
    m = matrix.astype(int)
    return m.T @ m


def crosstab_ordered(df, index, columns, index_order, columns_order) -> pd.DataFrame:
    """Contingency table with fixed category order; only groups that occur are kept as rows."""
    tab = pd.crosstab(df[index], df[columns])
    rows = [i for i in index_order if i in tab.index]
    return tab.reindex(index=rows, columns=list(columns_order), fill_value=0)


def ordinal_scores(series: pd.Series, order) -> pd.Series:
    return series.map({value: i for i, value in enumerate(order)})


def constraint_coverage(df: pd.DataFrame, sc: S.Schema) -> pd.DataFrame:
    """Per constraint: how many respondents selected it with / without a written explanation."""
    rows = {}
    for label, col in sc.constraints.items():
        selected = df[col].eq("Yes")
        has_comment = df[sc.constraint_comments[label]].fillna("").str.strip().ne("")
        rows[label] = {"Selected, with explanation": int((selected & has_comment).sum()),
                       "Selected, no explanation": int((selected & ~has_comment).sum())}
    return pd.DataFrame(rows).T

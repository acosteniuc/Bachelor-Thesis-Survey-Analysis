"""Codebooks for the open-text questions plus helpers to read the manual coding (data/coding/*.csv)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import schema as S
from .plots import NAVY, ORANGE, TEAL, GREY

CODING_DIR = Path(__file__).resolve().parent.parent / "data" / "coding"


@dataclass(frozen=True)
class Code:
    name: str
    group: str
    definition: str


PROCESS_FACTORS = [
    Code("Standards & regulations", "External requirements",
         "Norms, standards or legal/regulatory frameworks prescribe or shape the process (e.g. ECSS, ESA/NASA standards)."),
    Code("Customer & partner requirements", "External requirements",
         "Customers, agencies or cooperation partners demand, approve or constrain the process."),
    Code("Schedule & time pressure", "Project context", "Deadlines and tight schedules drive process choices."),
    Code("Mission & project characteristics", "Project context",
         "Mission type, mission requirements or project lifetime (e.g. New Space vs. old space)."),
    Code("Quality & security needs", "Project context",
         "Reliability, reproducibility, efficiency, security or data privacy as drivers of the process."),
    Code("Team experience & familiarity", "People & tools",
         "The team's prior experience or the time needed to become familiar with a process."),
    Code("Personal preference & social influence", "People & tools",
         "Preferences of leads/owners, recommendations of colleagues, individual attitudes."),
    Code("Tooling & infrastructure", "People & tools", "Available tools and tool support for the process."),
]

SIMILAR_REASON = [
    Code("Technology & practice transfer", "Pushes towards similarity",
         "Tools, platforms or working practices from other domains (CI/CD, simulation, better SE workflows) arrive in space."),
    Code("Standardization & interoperability", "Pushes towards similarity",
         "Common standards, platforms or interfaces reduce domain-specific solutions."),
    Code("AI & automation", "Pushes towards similarity",
         "AI-assisted coding and test automation change both worlds in the same way."),
    Code("Commercialization (New Space)", "Pushes towards similarity",
         "New Space companies bring commercial software engineering into the space sector."),
    Code("Cost & market pressure", "Pushes towards similarity",
         "Economic pressure and affordability push space projects towards cheaper, common approaches."),
    Code("Openness & culture change", "Pushes towards similarity",
         "Open source, public accessibility and cultural opening of space software."),
    Code("Shared challenges", "Pushes towards similarity",
         "Both sectors face the same quality, reliability and security challenges."),
    Code("Criticality & strict standards", "Pushes towards difference",
         "Safety/mission criticality and strict QA standards keep space engineering distinct."),
    Code("Space-specific constraints", "Pushes towards difference",
         "Harsh environment and embedded, mission-specific hardware conditions."),
    Code("Security & sovereignty", "Pushes towards difference",
         "National security and sovereignty concerns limit what space can adopt."),
    Code("Limited applicability of AI", "Pushes towards difference",
         "AI tools/cloud services or training data are not applicable in the space domain."),
    Code("Conservatism & slow change", "Inertia", "The sector evolves slowly or resists modernization."),
]

CONSTRAINT_COMMENTS = [
    Code("Hardware & resource limits", "Technical",
         "Limited CPU, memory, GPUs, peripherals or hardware-bound development."),
    Code("Constrained updates & maintenance", "Technical",
         "Software cannot be updated after deployment, updates are slow, or downtime is not allowed."),
    Code("Availability & long-term operation", "Technical",
         "Long mission lifetimes, high availability targets and long-term maintainability."),
    Code("Data volume & scalability", "Technical", "Very large data volumes shape the architecture."),
    Code("Standards & regulatory compliance", "Compliance",
         "ECSS, DO-178, BSI Grundschutz, criticality levels or other regulation must be fulfilled."),
    Code("Security requirements", "Compliance", "Security regulation, military context or data protection needs."),
    Code("Budget & manpower limits", "Project & organization", "Small budgets, limited resources or few people."),
    Code("Schedule & time pressure", "Project & organization",
         "Fixed launch dates, short preparation time, tight planning or backlogs."),
    Code("Legacy & reuse", "Project & organization",
         "Legacy hardware/software, long-maintained in-house libraries, reuse across missions."),
    Code("Organizational & people factors", "Project & organization",
         "Suppliers, availability of staff, house policies, routines and decision making."),
]

CODEBOOKS = {
    "process_factors": PROCESS_FACTORS,
    "similar_reason": SIMILAR_REASON,
    "constraint_comments": CONSTRAINT_COMMENTS,
}

GROUP_COLORS = {
    "process_factors": {"External requirements": NAVY, "Project context": TEAL, "People & tools": ORANGE},
    "similar_reason": {"Pushes towards similarity": "#1F6FA8", "Pushes towards difference": "#B2501A", "Inertia": GREY},
    "constraint_comments": {"Technical": NAVY, "Compliance": "#8E6BBF", "Project & organization": ORANGE},
}

FIELD_TITLES = {
    "process_factors": "Factors influencing the choice of the development process",
    "similar_reason": "Reasons for expecting similar / different software engineering",
    "constraint_comments": "Explanations of the selected project constraints",
    "anything_else": "Additional remarks",
}


def codebook_frame(field: str) -> pd.DataFrame:
    return pd.DataFrame([vars(c) for c in CODEBOOKS[field]]).rename(
        columns={"name": "Theme", "group": "Group", "definition": "Definition"})


def theme_groups(field: str) -> dict:
    return {c.name: c.group for c in CODEBOOKS[field]}


def theme_names(field: str) -> list[str]:
    return [c.name for c in CODEBOOKS[field]]


# ---------------------------------------------------------------- texts and codes
def _texts(df: pd.DataFrame, sc: S.Schema, col: str) -> pd.DataFrame:
    block = df[[sc.response_id, col]].dropna()
    text = block[col].astype(str).str.strip()
    keep = text != ""
    return pd.DataFrame({"response_id": block.loc[keep, sc.response_id].astype(int), "text": text[keep]}
                        ).reset_index(drop=True)


def open_text(df: pd.DataFrame, sc: S.Schema, field: str) -> pd.DataFrame:
    """Non-empty free-text answers: columns response_id, text (and constraint for constraint comments)."""
    if field == "constraint_comments":
        parts = []
        for label, col in sc.constraint_comments.items():
            part = _texts(df, sc, col)
            part.insert(1, "constraint", label)
            parts.append(part)
        return pd.concat(parts, ignore_index=True).sort_values(["response_id", "constraint"], ignore_index=True)
    column = {"process_factors": sc.process_factors, "similar_reason": sc.similar_comment,
              "anything_else": sc.anything_else}[field]
    return _texts(df, sc, column)


def read_coding(field: str) -> pd.DataFrame:
    """Long table (response_id, [constraint], code) from data/coding/<field>.csv; codes are ';'-separated."""
    path = CODING_DIR / f"{field}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["response_id", "code"])
    raw = pd.read_csv(path)
    keys = [c for c in raw.columns if c != "codes"]
    raw["code"] = raw["codes"].fillna("").str.split(";")
    long = raw.explode("code")
    long["code"] = long["code"].str.strip()
    return long.loc[long["code"] != "", keys + ["code"]].reset_index(drop=True)


def validate_coding(field: str, texts: pd.DataFrame) -> pd.DataFrame:
    """Checks the manual coding against the codebook and the texts; raises on unknown codes."""
    coded = read_coding(field)
    unknown = sorted(set(coded["code"]) - set(theme_names(field)))
    if unknown:
        raise ValueError(f"[{field}] codes not in codebook: {unknown}")
    keys = [c for c in texts.columns if c != "text"]
    text_keys = set(map(tuple, texts[keys].values.tolist()))
    coded_keys = set(map(tuple, coded[keys].drop_duplicates().values.tolist())) if len(coded) else set()
    uncoded, orphan = sorted(text_keys - coded_keys), sorted(coded_keys - text_keys)
    print(f"[{field}] {len(text_keys)} answers, {len(coded_keys & text_keys)} coded, "
          f"{len(uncoded)} uncoded, {len(orphan)} codes without answer")
    if uncoded:
        print("   uncoded:", uncoded)
    if orphan:
        print("   coded but no answer:", orphan)
    return coded


def theme_matrix(field: str, texts: pd.DataFrame) -> pd.DataFrame:
    """Boolean matrix (answer x theme) in codebook order."""
    coded = read_coding(field)
    keys = [c for c in texts.columns if c != "text"]
    index = pd.MultiIndex.from_frame(texts[keys]) if len(keys) > 1 else pd.Index(texts[keys[0]])
    flags = coded.groupby(keys + ["code"]).size().unstack("code", fill_value=0) > 0 if len(coded) else pd.DataFrame()
    return flags.reindex(index=index, columns=theme_names(field), fill_value=False).astype(bool)


def theme_counts(matrix: pd.DataFrame, field: str):
    """Theme frequencies (descending) and matching group colours for bar charts."""
    counts = matrix.sum().astype(int).sort_values(ascending=False, kind="stable")
    colors = [GROUP_COLORS[field][theme_groups(field)[t]] for t in counts.index]
    return counts, colors


def theme_by_group(matrix: pd.DataFrame, groups: pd.Series, group_order) -> pd.DataFrame:
    """Number of answers per theme (rows) within each respondent group (columns, labelled with group size)."""
    g = groups.reindex(matrix.index)
    sizes = g.value_counts()
    table = matrix.astype(int).groupby(g.values).sum().T
    table = table.reindex(columns=[x for x in group_order if x in table.columns])
    table.columns = [f"{c}\n(n={int(sizes[c])})" for c in table.columns]
    return table


def theme_by_constraint(coded: pd.DataFrame, field: str = "constraint_comments") -> pd.DataFrame:
    table = pd.crosstab(coded["code"], coded["constraint"])
    return table.reindex(index=theme_names(field), columns=list(S.CONSTRAINTS)[:-1], fill_value=0)


def review_table(texts: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Verbatim answers next to the assigned themes (for the appendix / manual review)."""
    out = texts.copy()
    out["themes"] = ["; ".join(matrix.columns[row]) for row in matrix.to_numpy()]
    return out


def quotes_by_theme(texts: pd.DataFrame, matrix: pd.DataFrame, per_theme: int = 2, max_chars: int = 260) -> pd.DataFrame:
    rows = []
    for theme in matrix.columns:
        hits = texts[matrix[theme].to_numpy()]
        for _, row in hits.head(per_theme).iterrows():
            text = row["text"].replace("\n", " ")
            entry = {"Theme": theme, "Respondent": row["response_id"]}
            if "constraint" in texts.columns:
                entry["Constraint"] = row["constraint"]
            entry["Quote"] = text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."
            rows.append(entry)
    return pd.DataFrame(rows)

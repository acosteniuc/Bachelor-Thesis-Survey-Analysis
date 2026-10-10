"""Survey schema: resolves the long LimeSurvey column names; English labels, orders and groupings."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


def norm(value):
    """Unify apostrophes and blanks (incl. non-breaking spaces); non-strings pass through unchanged."""
    if not isinstance(value, str):
        return value
    value = value.replace("\xa0", " ").replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"[ \t]+", " ", value).strip()


def _find(columns, *needles, endswith=None):
    hits = [
        c for c in columns
        if all(n in c for n in needles) and (endswith is None or c.endswith(endswith))
    ]
    if len(hits) != 1:
        raise KeyError(f"Expected exactly one column for {needles} (endswith={endswith!r}), found {len(hits)}")
    return hits[0]


# label used in figures -> fragment of the column header
LEARN_SOURCES = {
    "Technical documentation": "Technical documentation",
    "Other online resources": "Other online resources",
    "Stack Overflow / Stack Exchange": "Stack Overflow",
    "Videos": "Videos",
    "AI code-generation tools": "AI CodeGen",
    "Blogs / podcasts": "Blogs or podcasts",
    "Online courses / certification": "Online Courses",
    "Books / physical media": "Books",
    "Colleagues / on-the-job training": "Colleague",
    "School (university, college)": "School",
    "Games / coding challenges": "Games",
    "Coding bootcamp": "Coding Bootcamp",
}

NFRS = {
    "Reliability": "Reliability",
    "Safety": "Safety",
    "Security / cybersecurity": "Security / Cybersecurity",
    "Performance": "Performance",
    "Maintainability": "Maintainability",
    "Availability": "Availability",
    "Usability": "Usability",
    "Don't know": "I don't know",
}

CONSTRAINTS = {
    "Hardware limitations": "Hardware limitations",
    "Long system / mission lifetime": "Long system or mission lifetime",
    "Limited maintenance / updates": "Limited opportunities for maintenance or updates",
    "Certification / regulation": "Certification or regulatory requirements",
    "Development costs": "Development costs",
    "Schedule constraints": "Schedule constraints",
    "Legacy systems": "Legacy systems",
    "Organizational constraints": "Organizational constraints",
    "Don't know": "I don't know",
}


@dataclass(frozen=True)
class Schema:
    response_id: str
    submitted: str
    last_page: str
    education: str
    years_coding: str
    ai_learning: str
    role: str
    years_pro: str
    languages: str
    ai_use: str
    ai_stance: str
    ai_threat: str
    flexibility: str
    process_factors: str
    similar_choice: str
    similar_comment: str
    anything_else: str
    learn: dict
    nfr: dict
    constraints: dict
    constraint_comments: dict

    def question_columns(self) -> list[str]:
        """All substantive question columns in questionnaire order."""
        cols = [self.education, self.years_coding, *self.learn.values(), self.ai_learning, self.role,
                self.years_pro, self.languages, self.ai_use, self.ai_stance, self.ai_threat, *self.nfr.values()]
        for label in self.constraints:
            cols += [self.constraints[label], self.constraint_comments[label]]
        return cols + [self.flexibility, self.process_factors, self.similar_choice,
                       self.similar_comment, self.anything_else]


def resolve(columns) -> Schema:
    cols = [norm(c) for c in columns]
    return Schema(
        response_id=_find(cols, "Response ID"),
        submitted=_find(cols, "Date submitted"),
        last_page=_find(cols, "Last page"),
        education=_find(cols, "highest level of formal education"),
        years_coding=_find(cols, "years have you been coding in total"),
        ai_learning=_find(cols, "learning AI programming or AI-enabled tooling"),
        role=_find(cols, "describes your current job"),
        years_pro=_find(cols, "years of professional work experience"),
        languages=_find(cols, "programming, scripting, and markup languages"),
        ai_use=_find(cols, "currently use AI tools"),
        ai_stance=_find(cols, "How favorable is your stance"),
        ai_threat=_find(cols, "Do you believe AI is a threat"),
        flexibility=_find(cols, "choose or adapt the software development process"),
        process_factors=_find(cols, "greatest influence on the choice or adaption"),
        similar_choice=_find(cols, "become more similar or more different", endswith="your answer."),
        similar_comment=_find(cols, "become more similar or more different", endswith="[Comment]"),
        anything_else=_find(cols, "anything else you would like to share"),
        learn={k: _find(cols, f"learn to code in the past year? [{frag}") for k, frag in LEARN_SOURCES.items()},
        nfr={k: _find(cols, "non-functional requirements", endswith=f"[{frag}]") for k, frag in NFRS.items()},
        constraints={k: _find(cols, "operational constraints", endswith=f"[{frag}]") for k, frag in CONSTRAINTS.items()},
        constraint_comments={k: _find(cols, "operational constraints", endswith=f"[{frag}][Comment]")
                             for k, frag in CONSTRAINTS.items()},
    )


# ---------------------------------------------------------------- value labels and orders
EDUCATION_LEVELS = [
    ("Primary", "Primary school"),
    ("Secondary", "Secondary school"),
    ("Some college", "Some college, no degree"),
    ("Associate", "Associate degree"),
    ("Bachelor", "Bachelor's degree"),
    ("Master", "Master's degree"),
    ("Professional", "Professional degree / PhD"),
    ("Something else", "Other"),
]
EDUCATION_ORDER = [label for _, label in EDUCATION_LEVELS]


def education_label(value):
    if not isinstance(value, str):
        return value
    for prefix, label in EDUCATION_LEVELS:
        if value.startswith(prefix):
            return label
    return value


YEARS_ORDER = ["Less than 1 year", "1 to 5 years", "6 to 10 years", "11 to 15 years", "16 to 20 years",
               "21 to 30 years", "31 to 40 years", "41 to 50 years", "More than 50 years"]


def years_label(value):
    if not isinstance(value, str):
        return value
    return value.replace(" to ", "–").replace(" years", "").replace(" year", "")


def pro_exp_group(value):
    if not isinstance(value, str):
        return pd.NA
    if value in ("Less than 1 year", "1 to 5 years", "6 to 10 years"):
        return "Up to 10 years"
    if value in ("11 to 15 years", "16 to 20 years"):
        return "11–20 years"
    return "More than 20 years"


PRO_EXP_GROUPS = ["Up to 10 years", "11–20 years", "More than 20 years"]

AI_USE_ORDER = ["Yes, I use AI tools daily", "Yes, I use AI tools weekly",
                "Yes, I use AI tools monthly or infrequently", "No, but I plan to soon", "No, and I don't plan to"]
AI_USE_LABELS = {AI_USE_ORDER[0]: "Daily", AI_USE_ORDER[1]: "Weekly", AI_USE_ORDER[2]: "Monthly or less",
                 AI_USE_ORDER[3]: "Not yet, but planned", AI_USE_ORDER[4]: "No, not planned"}
AI_USE_GROUPS = ["Daily / weekly", "Occasional", "Non-user"]
_AI_USE_GROUP = {AI_USE_ORDER[0]: AI_USE_GROUPS[0], AI_USE_ORDER[1]: AI_USE_GROUPS[0],
                 AI_USE_ORDER[2]: AI_USE_GROUPS[1], AI_USE_ORDER[3]: AI_USE_GROUPS[2],
                 AI_USE_ORDER[4]: AI_USE_GROUPS[2]}


def ai_use_group(value):
    return _AI_USE_GROUP.get(value, pd.NA)


AI_STANCE_ORDER = ["Very unfavorable", "Unfavorable", "Indifferent", "Unsure", "Favorable", "Very favorable"]
AI_STANCE_GROUPS = ["Unfavorable", "Neutral / unsure", "Favorable"]
_AI_STANCE_GROUP = {"Very unfavorable": "Unfavorable", "Unfavorable": "Unfavorable",
                    "Indifferent": "Neutral / unsure", "Unsure": "Neutral / unsure",
                    "Favorable": "Favorable", "Very favorable": "Favorable"}


def ai_stance_group(value):
    return _AI_STANCE_GROUP.get(value, pd.NA)


AI_THREAT_ORDER = ["No", "I'm not sure", "Yes"]

AI_LEARNING_ORDER = ["AI skills – job / career", "AI skills – personal interest",
                     "Other skills – job / career", "Other skills – personal interest", "Did not learn anything"]


def ai_learning_label(value):
    if not isinstance(value, str):
        return value
    if value.startswith("No, I didn't"):
        return AI_LEARNING_ORDER[4]
    topic = "AI skills" if value.startswith("Yes") else "Other skills"
    purpose = "personal interest" if "personal" in value else "job / career"
    return f"{topic} – {purpose}"


FLEXIBILITY_ORDER = ["No flexibility", "Limited flexibility", "Moderate flexibility", "High flexibility",
                     "Complete flexibility", "I don't know"]


def flexibility_label(value):
    return value.replace(" flexibility", "") if isinstance(value, str) else value


FLEXIBILITY_GROUPS = ["Limited", "Moderate", "High or complete"]
_FLEX_GROUP = {"No flexibility": "Limited", "Limited flexibility": "Limited", "Moderate flexibility": "Moderate",
               "High flexibility": "High or complete", "Complete flexibility": "High or complete"}


def flexibility_group(value):
    return _FLEX_GROUP.get(value, pd.NA)


SIMILAR_ORDER = ["Become much more different", "Become somewhat more different",
                 "Remain approximately as they are today", "Become somewhat more similar", "Become much more similar"]
SIMILAR_LABELS = {SIMILAR_ORDER[0]: "Much more different", SIMILAR_ORDER[1]: "Somewhat more different",
                  SIMILAR_ORDER[2]: "About the same", SIMILAR_ORDER[3]: "Somewhat more similar",
                  SIMILAR_ORDER[4]: "Much more similar"}
SIMILAR_GROUPS = ["More different", "About the same", "More similar"]
_SIMILAR_GROUP = {SIMILAR_ORDER[0]: SIMILAR_GROUPS[0], SIMILAR_ORDER[1]: SIMILAR_GROUPS[0],
                  SIMILAR_ORDER[2]: SIMILAR_GROUPS[1], SIMILAR_ORDER[3]: SIMILAR_GROUPS[2],
                  SIMILAR_ORDER[4]: SIMILAR_GROUPS[2]}


def similar_group(value):
    return _SIMILAR_GROUP.get(value, pd.NA)


ROLE_GROUPS = ["Software engineering", "Research & science", "Management", "Other"]


def role_group(value):
    if not isinstance(value, str):
        return pd.NA
    v = value.lower()
    if "manager" in v:
        return "Management"
    if "researcher" in v or "scientist" in v:
        return "Research & science"
    if v.startswith("developer") or "devops" in v or "cybersecurity" in v:
        return "Software engineering"
    return "Other"


def role_label(value):
    if not isinstance(value, str):
        return value
    match = re.match(r"Developer, (.+)", value)
    if match:
        inner = (match.group(1).replace("embedded applications or devices", "embedded")
                 .replace("desktop or enterprise applications", "desktop / enterprise")
                 .replace("QA or test", "QA / test"))
        return f"Developer ({inner})"
    return (value.replace("DevOps engineer or professional", "DevOps engineer")
            .replace("Cybersecurity or InfoSec professional", "Cybersecurity professional"))

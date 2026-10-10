"""Figure helpers: one consistent style and the chart types used for the thesis figures (English labels)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation

FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
SHOW_TITLES = True  # set to False when the LaTeX caption replaces the figure title

NAVY, ORANGE, TEAL, GREY, GRID, INK = "#1F4E79", "#D9822B", "#2A9D8F", "#8D99AE", "#E5E8EC", "#2B2D42"
STANCE_COLORS = ["#B2501A", "#E8A06A", "#CFCFCF", "#EBEBEB", "#8FBBD9", "#1F6FA8"]  # unfavorable -> favorable
DIVERGING5 = ["#B2501A", "#E8A06A", "#D3D3D3", "#8FBBD9", "#1F6FA8"]
GROUP_PALETTE = [NAVY, ORANGE, TEAL, "#8E6BBF", "#B8860B", GREY]


def setup_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "pdf.fonttype": 42,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#B8BEC7", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.frameon": False, "legend.fontsize": 9,
    })


setup_style()


def sequential(n: int) -> list:
    return list(sns.blend_palette(["#D6E4F0", NAVY], n)) if n > 1 else [NAVY]


FIGURE_FOLDERS = {  # figure name -> sub-folder (one folder per survey question, in questionnaire order)
    "response_flow": "00_overview", "ordinal_correlations": "00_overview",
    "sample_education": "01_education",
    "sample_experience": "02_experience",
    "learning_sources": "03_learning_sources",
    "learning_ai_at_work": "04_ai_learning_at_work",
    "sample_role": "05_role",
    "sample_language": "06_programming_language",
    "ai_usage": "07_ai_usage",
    "ai_stance": "08_ai_stance", "ai_stance_by_usage": "08_ai_stance", "ai_stance_by_role": "08_ai_stance",
    "ai_stance_by_experience": "08_ai_stance",
    "ai_threat": "09_ai_threat", "ai_threat_by_stance": "09_ai_threat",
    "nfr_ranking": "10_non_functional_requirements", "nfr_cooccurrence": "10_non_functional_requirements",
    "constraints_ranking": "11_project_constraints", "constraints_cooccurrence": "11_project_constraints",
    "constraints_vs_nfr": "11_project_constraints",
    "open_constraints_coverage": "11b_constraint_explanations_OPEN",
    "open_constraints_themes": "11b_constraint_explanations_OPEN",
    "open_constraints_theme_by_constraint": "11b_constraint_explanations_OPEN",
    "process_flexibility": "12_process_flexibility", "flexibility_by_certification": "12_process_flexibility",
    "flexibility_by_role": "12_process_flexibility",
    "open_process_factors_themes": "13_process_factors_OPEN",
    "open_process_factors_oncoprint": "13_process_factors_OPEN",
    "open_process_factors_by_role": "13_process_factors_OPEN",
    "open_process_factors_by_flexibility": "13_process_factors_OPEN",
    "similar_or_different": "14_similar_or_different", "similar_or_different_by_ai_stance": "14_similar_or_different",
    "similar_or_different_by_role": "14_similar_or_different",
    "open_similar_reason_themes": "14b_similar_or_different_reasons_OPEN",
    "open_similar_reason_oncoprint": "14b_similar_or_different_reasons_OPEN",
    "open_similar_reason_by_expectation": "14b_similar_or_different_reasons_OPEN",
    "open_similar_reason_direction": "14b_similar_or_different_reasons_OPEN",
    "open_similar_reason_cooccurrence": "14b_similar_or_different_reasons_OPEN",
    "open_answer_length": "15_open_text_overview", "open_participation_role_group": "15_open_text_overview",
    "open_participation_ai_stance": "15_open_text_overview", "open_keyword_prevalence": "15_open_text_overview",
    "open_top_terms": "15_open_text_overview", "open_top_bigrams": "15_open_text_overview",
    "open_sentiment": "15_open_text_overview",
}


def save_figure(fig: plt.Figure, name: str, formats=("png", "pdf")) -> Path:
    """Saves PNG (300 dpi) and vector PDF into outputs/figures/<question folder>/."""
    folder = FIGURES_DIR / FIGURE_FOLDERS.get(name, "")
    folder.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(folder / f"{name}.{fmt}", dpi=300, bbox_inches="tight")
    return folder / f"{name}.png"


# ---------------------------------------------------------------- internals
def _wrap(label, width: int = 30) -> str:
    return textwrap.fill(str(label), width, break_long_words=False)


def _height(rows: int, per_row: float = 0.4, base: float = 1.3) -> float:
    return base + per_row * rows


def _prepare(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def _header(ax, title: str, n=None) -> None:
    if n is not None:
        ax.text(0, 1.01, f"n = {n}", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=GREY)
    if SHOW_TITLES and title:
        ax.set_title(title, pad=18 if n is not None else 8)


def _ink_on(color) -> str:
    r, g, b = mpl.colors.to_rgb(color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def _count_label(value, n, show_percent: bool) -> str:
    return f"{int(value)} ({value / n:.0%})" if show_percent and n else f"{int(value)}"


def _legend_below(ax, ncol: int, offset_in: float = 0.55, **kwargs):
    shift = ScaledTranslation(0, -offset_in, ax.figure.dpi_scale_trans)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=ax.transAxes + shift,
                     ncol=ncol, **kwargs)


def add_legend(ax, mapping: dict, loc: str = "lower right", title=None) -> None:
    """Colour legend for bars that are coloured by a category (label -> colour)."""
    ax.legend(handles=[Patch(color=c, label=l) for l, c in mapping.items()], loc=loc, title=title,
              alignment="left")


# ---------------------------------------------------------------- basic charts
def bar_chart(counts: pd.Series, title: str = "", n=None, horizontal: bool = True, colors=None,
              xlabel: str = "Number of respondents", show_percent: bool = True, wrap: int = 30,
              ax=None, figsize=None) -> plt.Figure:
    """Bar chart of counts with value labels; `n` is the denominator for the percentages."""
    values = counts.astype(float).values
    n = int(n if n is not None else values.sum())
    k, top = len(counts), (values.max() if len(values) and values.max() > 0 else 1)
    colors = colors if colors is not None else NAVY
    pos = np.arange(k)
    if horizontal:
        fig, ax = _prepare(ax, figsize or (6.4, _height(k)))
        ax.barh(pos, values, color=colors, height=0.68)
        ax.set_yticks(pos, [_wrap(i, wrap) for i in counts.index])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, top * 1.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.grid(True, color=GRID)
        ax.tick_params(axis="y", length=0)
        for p, v in zip(pos, values):
            ax.text(v + top * 0.015, p, _count_label(v, n, show_percent), va="center", fontsize=9)
    else:
        fig, ax = _prepare(ax, figsize or (max(4.8, 1.0 + 0.75 * k), 3.6))
        ax.bar(pos, values, color=colors, width=0.68)
        ax.set_xticks(pos, [_wrap(i, 14) for i in counts.index])
        ax.set_ylabel(xlabel)
        ax.set_ylim(0, top * 1.2)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.grid(True, color=GRID)
        ax.tick_params(axis="x", length=0)
        for p, v in zip(pos, values):
            ax.text(p, v + top * 0.02, _count_label(v, n, show_percent), ha="center", va="bottom", fontsize=9)
    _header(ax, title, n)
    return fig


def lollipop_chart(counts: pd.Series, n: int, title: str = "", colors=None, sort: bool = True,
                   xlabel: str = "Share of respondents (%)", wrap: int = 34, ax=None, figsize=None) -> plt.Figure:
    """Share of respondents per option (multi-select questions)."""
    shares = counts / n * 100
    if sort:
        order = shares.sort_values(ascending=False, kind="stable").index
        shares, counts = shares[order], counts[order]
        if isinstance(colors, dict):
            colors = [colors[i] for i in order]
    colors = colors if colors is not None else NAVY
    k, pos = len(shares), np.arange(len(shares))
    fig, ax = _prepare(ax, figsize or (6.6, _height(k)))
    ax.hlines(pos, 0, shares.values, color="#B8C4D6", lw=2.2)
    ax.scatter(shares.values, pos, s=75, color=colors, zorder=3)
    ax.set_yticks(pos, [_wrap(i, wrap) for i in shares.index])
    ax.invert_yaxis()
    ax.set_xlim(0, min(100, shares.max() * 1.3 + 6))
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID)
    ax.tick_params(axis="y", length=0)
    for p, s, c in zip(pos, shares.values, counts.values):
        ax.text(s + 1.6, p, f"{s:.0f}% ({int(c)})", va="center", fontsize=9)
    _header(ax, title, n)
    return fig


def donut_chart(counts: pd.Series, colors, title: str = "", ax=None, figsize=(5.6, 3.4)) -> plt.Figure:
    n = int(counts.sum())
    fig, ax = _prepare(ax, figsize)
    wedges, _ = ax.pie(counts.values, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"n = {n}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.legend(wedges, [f"{label}: {int(v)} ({v / n:.0%})" for label, v in counts.items()],
              loc="center left", bbox_to_anchor=(1.0, 0.5))
    if SHOW_TITLES and title:
        ax.set_title(title, loc="center")
    return fig


def stacked_bars(counts: pd.DataFrame, colors, title: str = "", normalize: bool = True, ax=None, figsize=None,
                 xlabel=None, min_label: float = 0.04, show_n: bool = True) -> plt.Figure:
    """Horizontal stacked bars; rows = groups, columns = answer categories (labels show counts)."""
    counts = counts.astype(float)
    totals = counts.sum(axis=1)
    data = counts.div(totals, axis=0) * 100 if normalize else counts
    k, span = len(data), (100 if normalize else totals.max())
    fig, ax = _prepare(ax, figsize or (7.4, 1.35 + 0.55 * k))
    left, rows = np.zeros(k), np.arange(k)
    for j, col in enumerate(data.columns):
        vals = data[col].values
        ax.barh(rows, vals, left=left, color=colors[j], edgecolor="white", linewidth=1.2, height=0.72, label=col)
        for i in range(k):
            if counts.iloc[i, j] > 0 and vals[i] >= min_label * span:
                ax.text(left[i] + vals[i] / 2, i, f"{int(counts.iloc[i, j])}", ha="center", va="center",
                        fontsize=8.5, color=_ink_on(colors[j]))
        left += vals
    ax.set_yticks(rows, [f"{_wrap(i, 26)}\n(n={int(t)})" if k > 1 else "" for i, t in totals.items()])
    ax.invert_yaxis()
    ax.set_xlim(0, span)
    if not normalize:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel or ("Share of respondents (%)" if normalize else "Number of respondents"))
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _legend_below(ax, ncol=len(data.columns) if len(data.columns) <= 3 else 3)
    _header(ax, title, int(totals.sum()) if k == 1 and show_n else None)
    return fig


def diverging_bars(counts: pd.DataFrame, colors, n_left: int, n_center: int, title: str = "", ax=None,
                   figsize=None) -> plt.Figure:
    """Likert-style bars centred on the neutral categories; columns run from negative to positive."""
    counts = counts.astype(float)
    totals = counts.sum(axis=1)
    pct = counts.div(totals, axis=0) * 100
    left_sum = pct.iloc[:, :n_left].sum(axis=1)
    center_sum = pct.iloc[:, n_left:n_left + n_center].sum(axis=1)
    start = (-(left_sum + center_sum / 2)).values
    k, rows = len(pct), np.arange(len(pct))
    fig, ax = _prepare(ax, figsize or (7.6, 1.35 + 0.55 * k))
    extent = np.abs(start).max()
    pos = start.copy()
    for j, col in enumerate(pct.columns):
        vals = pct[col].values
        ax.barh(rows, vals, left=pos, color=colors[j], edgecolor="white", linewidth=1.2, height=0.72, label=col)
        for i in range(k):
            if counts.iloc[i, j] > 0 and vals[i] >= 4:
                ax.text(pos[i] + vals[i] / 2, i, f"{int(counts.iloc[i, j])}", ha="center", va="center",
                        fontsize=8.5, color=_ink_on(colors[j]))
        pos = pos + vals
    extent = max(extent, pos.max())
    lim = min(100, np.ceil((extent + 4) / 10) * 10)
    step = 20 if lim > 40 else 10
    reach = (lim // step) * step
    ax.axvline(0, color=INK, lw=0.9, zorder=0)
    ax.set_xlim(-lim, lim)
    ticks = np.arange(-reach, reach + 1, step)
    ax.set_xticks(ticks, [f"{abs(int(t))}" for t in ticks])
    ax.set_yticks(rows, [f"{_wrap(i, 26)}\n(n={int(t)})" if k > 1 else "" for i, t in totals.items()])
    ax.invert_yaxis()
    ax.set_xlabel("Share of respondents (%)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _legend_below(ax, ncol=3)
    _header(ax, title, int(totals.sum()) if k == 1 else None)
    return fig


def grouped_bars(frame: pd.DataFrame, colors, title: str = "", ylabel: str = "Number of respondents", ax=None,
                 figsize=None) -> plt.Figure:
    """Vertical bars side by side; rows = categories, columns = series."""
    m, k = frame.shape[1], len(frame)
    x, width = np.arange(k), 0.8 / frame.shape[1]
    fig, ax = _prepare(ax, figsize or (max(5.6, 0.9 * k + 2), 3.8))
    top = max(frame.values.max(), 1)
    for j, col in enumerate(frame.columns):
        offset = (j - (m - 1) / 2) * width
        ax.bar(x + offset, frame[col].values, width * 0.92, color=colors[j], label=f"{col} (n={int(frame[col].sum())})")
        for xi, v in zip(x + offset, frame[col].values):
            if v > 0:
                ax.text(xi, v + top * 0.015, f"{int(v)}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, [_wrap(i, 14) for i in frame.index])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, top * 1.2)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.grid(True, color=GRID)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper right")
    _header(ax, title)
    return fig


def heatmap(matrix: pd.DataFrame, title: str = "", cmap=None, fmt: str = "d", hide_zero: bool = True,
            mask_upper: bool = False, cbar_label=None, center=None, vmin=None, vmax=None, xlabel: str = "",
            ylabel: str = "", col_wrap: int = 16, row_wrap: int = 34, cbar: bool = True, n=None, ax=None,
            figsize=None) -> plt.Figure:
    """Annotated heatmap for cross-tabulations, co-occurrence and association matrices."""
    rows, cols = matrix.shape
    fig, ax = _prepare(ax, figsize or (2.6 + 0.8 * cols, 1.4 + 0.42 * rows))

    def text(v):
        if pd.isna(v) or (hide_zero and v == 0):
            return ""
        return f"{int(round(v))}" if fmt == "d" else format(v, fmt)

    annot = matrix.map(text).values
    mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1) if mask_upper else None
    cmap = cmap or sns.light_palette(NAVY, as_cmap=True)
    sns.heatmap(matrix.astype(float), ax=ax, cmap=cmap, annot=annot, fmt="", mask=mask, linewidths=1.2,
                linecolor="white", center=center, vmin=vmin, vmax=vmax, cbar=cbar,
                cbar_kws={"label": cbar_label or "", "shrink": 0.8}, annot_kws={"fontsize": 9})
    if cbar and fmt == "d":
        colorbar = ax.collections[0].colorbar
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
    ax.set_xticklabels([_wrap(c, col_wrap) for c in matrix.columns], rotation=0 if cols <= 6 else 40,
                       ha="center" if cols <= 6 else "right")
    ax.set_yticklabels([_wrap(r, row_wrap) for r in matrix.index], rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _header(ax, title, n)
    return fig


def oncoprint(matrix: pd.DataFrame, col_groups=None, group_colors=None, group_order=None, row_groups=None,
              row_group_colors=None, title: str = "", strip_title: str = "Respondent group",
              ax=None, figsize=None) -> plt.Figure:
    """Dot matrix of coded themes (rows) per respondent (columns); one glance shows every coding decision."""
    m = matrix.astype(bool)
    ids = list(m.index)
    groups = col_groups.reindex(ids) if col_groups is not None else pd.Series("All", index=ids)
    order = group_order or list(dict.fromkeys(groups.dropna()))
    rank = {g: i for i, g in enumerate(order)}
    sizes = m.sum(axis=1)
    ids.sort(key=lambda i: (rank.get(groups[i], len(order)), -sizes[i], i))
    themes = list(m.columns)
    k, nc = len(themes), len(ids)
    fig, ax = _prepare(ax, figsize or (2.8 + 0.34 * nc + 2.2, 1.6 + 0.4 * k))
    group_colors = group_colors or {g: GROUP_PALETTE[i % len(GROUP_PALETTE)] for i, g in enumerate(order)}
    for j, rid in enumerate(ids):
        ax.add_patch(Rectangle((j - 0.4, -1.5), 0.8, 0.7, color=group_colors.get(groups[rid], GREY)))
        for i, theme in enumerate(themes):
            on = m.loc[rid, theme]
            color = (row_group_colors or {}).get((row_groups or {}).get(theme), NAVY) if on else "#EDF0F4"
            ax.add_patch(Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, color=color))
    ax.set_xlim(-0.6, nc - 0.4)
    ax.set_ylim(k - 0.4, -2.0)
    ax.set_xticks(range(nc), [str(i) for i in ids], rotation=90, fontsize=8.5)
    ax.set_yticks(range(k), [f"{_wrap(t, 34)}  ({int(m[t].sum())})" for t in themes])
    ax.set_xlabel("Respondent ID")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legends = []
    if col_groups is not None:
        legends.append((strip_title, [Patch(color=group_colors.get(g, GREY), label=g) for g in order]))
    if row_groups and row_group_colors:
        used = list(dict.fromkeys(row_groups[t] for t in themes if t in row_groups))
        legends.append(("Theme group", [Patch(color=row_group_colors[g], label=g) for g in used]))
    pos = ax.get_position()
    x, y = pos.x1 + 0.15 / fig.get_figwidth(), pos.y1
    for heading, handles in legends:
        fig.legend(handles=handles, title=heading, loc="upper left", bbox_to_anchor=(x, y), alignment="left",
                   title_fontsize=9)
        y -= 0.28 * (len(handles) + 2) / fig.get_figheight()
    _header(ax, title, nc)
    return fig


def strip_box(data: pd.DataFrame, value: str, category: str, order=None, title: str = "", xlabel: str = "",
              ref_lines=(), ax=None, figsize=None) -> plt.Figure:
    """Box plot with all individual points overlaid (suited for very small samples)."""
    order = order or list(data[category].dropna().unique())
    fig, ax = _prepare(ax, figsize or (6.8, _height(len(order), 0.7, 1.2)))
    sns.boxplot(data=data, x=value, y=category, order=order, orient="h", color="#E9EEF4", width=0.5,
                fliersize=0, linewidth=1, ax=ax)
    sns.stripplot(data=data, x=value, y=category, order=order, orient="h", color=NAVY, size=5.5, jitter=0.14,
                  alpha=0.9, ax=ax)
    counts = data.groupby(category)[value].count()
    ax.set_yticks(range(len(order)), [f"{_wrap(o, 28)}\n(n={int(counts.get(o, 0))})" for o in order])
    for x in ref_lines:
        ax.axvline(x, color=GREY, lw=0.9, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.xaxis.grid(True, color=GRID)
    ax.tick_params(axis="y", length=0)
    _header(ax, title)
    return fig

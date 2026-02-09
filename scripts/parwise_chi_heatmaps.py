"""
Generates chi-square tests and heatmap visualizations for
all pairwise combinations of categorical variables, exported to a PDF.

Original R implementation provided by Dr. Istvan David.
"""

__title__   = "Pairwise Chi-Square Heatmaps"
__author__  = "Feyi Adesanya"

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# CONFIGURATION
EXCEL_PATH = "data/Data extraction sheet.xlsx"
SHEET_NAME = "Sheet1"
HEADER_START_ROW = 1

# All columns to be included in the evaluation
WANTED_COLUMNS = [
    "Motivation (Clustered)",
    "Domain (Aggregated)",
    "Intent",
    "Constituent unit (higher level aggregation)",
    "SoTS Classification",
    "DT Class",
    "SoS: Autonomy of Constituents",
    "SoS: Independence",
    "SoS: Distribution",
    "SoS: Evolution",
    "SoS: Dynamic Reconfiguration",
    "SoS: Emergence of Behaviour",
    "SoS: Interdependence",
    "SoS: Interoperability",
    "Emergence",
    "Security/Confidentiality Level",
    "Reliability Level",
    "TRL",
    "Evaluation",
    "Eval/Val Expanded",
    "Contribution type",
    "Publication type",
    "Do The Studies Use Standards in More of an SoS or DT context",
    "Publisher",
    # MULTI VALUE COLUMNS
    "Services (Cleaned)",
    "Standards Used (Cleaned Up)",
    "Modeling and Simulation Formalism types",

]

# Multi-valued columns and their delimiters
MULTIVALUE_COLUMNS = {
    "Services (Cleaned)": ",",
    "Standards Used (Cleaned Up)": ";",
    "Modeling and Simulation Formalism types": ",",
}

# Output file location
PDF_OUTPUT = "output/contingency_tables/pairwise_chi_heatmaps.pdf"
PDF_OUTPUT_RESIDUALS = "output/contingency_tables/pairwise_chi_residuals.pdf"

# Title colour thresholds
P_THRESH_SIG = 0.05
P_THRESH_MARGINAL = 0.10
MIN_COVERAGE_RATIO = 0.90 

# Qualiy Gates
QUALITY_COLUMNS = [
    "Q1: SoS is clear",
    "Q2: DT is clear",
    "Q3: Tangible contributions",
    "Q4: Reporting clarity",
]

# Custom Ordering
orders = {
    "Reliability Level": ["Not Addressed", "Mentioned", "Architecturally Addressed", "Explicitly Modeled", "Evaluated or Validated"] ,
    "Security/Confidentiality Level":["Not Addressed", "Mentioned", "Architecturally Addressed", "Explicitly Modeled", "Evaluated or Validated"] ,
    "TRL": ["Initial", "Proof-of-Concept", "Demo Prototype", "Deployed Prototype", "Operational"],
    "Emergence": ["Not Addressed", "Simple", "Weak", "Strong"],
    "Contribution type": ["Conceptual", "Technical", "Case Study"],
    "Constituent unit (higher level aggregation)": ["Physical Systems", "Cyber Physical Systems", "Cyber-Physical-Human Systems", "Enterprise Systems"],
    "SoTS Classification": ["Directed SoTS", "Acknowledged SoTS", "Collaborative SoTS", "Virtual SoTS"],
    "Evaluation": ["Validation", "Evaluation"],
    "DT Class": ["Digital Model", "Digital Shadow", "Digital Twin", "Human-Actuated Digital Twin", "Human-Supervised Digital Twin"],
    "Eval/Val Expanded": ["Architectural/Conceptual Design", "Mathematical analysis", "Empirical Simulation", "Laboratory experiments", "Prototyping", "Action Research", "Industrial Case Study"],
    "SoS: Autonomy of Constituents": ["No", "Partial", "Yes"],
    "SoS: Independence": ["No", "Partial", "Yes"],
    "SoS: Distribution": ["No", "Partial", "Yes"],
    "SoS: Evolution": ["No", "Partial", "Yes"],
    "SoS: Emergence of Behaviour": ["No", "Partial", "Yes"],
    "SoS: Interdependence": ["No", "Partial", "Yes"],
    "SoS: Interoperability": ["No", "Partial", "Yes"],
    "SoS: Dynamic Reconfiguration": ["No", "Partial", "Yes"],
    "Publication type": ["Workshop", "Book chapter", "Conference", "Journal"]
}
# Uppercases all of them for consistency
for col in orders:
    orders[col] = [val.upper() for val in orders[col]]

frequency_colour_gradient = LinearSegmentedColormap.from_list(
    "freq",
    ["#FFFFFF", "#FFD27F", "#FF8C33", "#E53935"],
    N=256,
)

def load_and_filter_data(path, sheet, header_row):
    df = pd.read_excel(path, sheet_name=sheet, header=header_row, dtype="object")
    df[QUALITY_COLUMNS] = df[QUALITY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=QUALITY_COLUMNS)
    df = df[~(df[QUALITY_COLUMNS] == 0).any(axis=1)]
    return df


def uppercase_strings(df):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = (
                out[col]
                .astype("string")
                .str.strip()
                .str.upper()
            )
    return out


def _split_tokens(series, delim):
    return (
        series.astype("string")
        .apply(lambda x: [] if pd.isna(x) else [t.strip() for t in str(x).split(delim) if t.strip()])
    )


def make_crosstab(df, col1, col2, row_id = "ROW_ID"):
    # Filter to the two specified columns
    sub = df[[row_id, col1, col2]].copy()

    for col in (col1, col2):
        if col in MULTIVALUE_COLUMNS and col in sub.columns:
            sub[col] = _split_tokens(sub[col], MULTIVALUE_COLUMNS[col])
            sub = sub.explode(col, ignore_index=True)
        # if a custom ordering is defined for this column
        if col in orders:
            sub[col] = pd.Categorical(sub[col], categories=orders[col], ordered=True)

    sub = sub.dropna(subset=[col1, col2])
    sub = sub.drop_duplicates([row_id, col1, col2])

    ct = pd.crosstab(sub[col1], sub[col2])
    return ct


def chi_square_with_and_without_yates(table):
    chi2, p, dof, exp = chi2_contingency(table, correction=False)
    chi2_y, p_y, dof_y, exp_y = chi2_contingency(table, correction=True)
    return chi2, p, dof, exp, chi2_y, p_y, dof_y, exp_y


def shorten_label(label, max_words=3):
    words = str(label).split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + " ..."
    return label


def draw_heatmap(ax, table, title, title_colour, residual_heatmap=False):
    data = table.values
    nrows, ncols = data.shape

    if residual_heatmap:
        color_data = np.abs(data)
    else:
        color_data = data

    im = ax.imshow(
        color_data,
        aspect="equal",
        vmin=0,
        vmax=color_data.max() if color_data.size else 1,
        cmap=frequency_colour_gradient,
    )

    # Colourbar for frequency
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.25)
    cbar = plt.colorbar(im, cax=cax, orientation="vertical")
    cbar.set_label("Frequency")
    if residual_heatmap:
        cbar.set_label("Residual")
    else:
        cbar.set_label("Frequency")

    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")

    ax.set_yticks(range(nrows))
    ax.set_yticklabels([shorten_label(l) for l in table.index.tolist()], fontsize=8)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([shorten_label(l) for l in table.columns.tolist()], rotation=45, ha="right", fontsize=8)

    # Cell labels
    for i in range(nrows):
        for j in range(ncols):
            # ax.text(j, i, str(int(data[i, j])), ha="center", va="center")
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xlabel(table.columns.name or "")
    ax.set_ylabel(table.index.name or "")
    ax.set_title(title, color=title_colour, pad=12)
    ax.grid(False)


def main():
    df_raw = load_and_filter_data(EXCEL_PATH, SHEET_NAME, HEADER_START_ROW)

    # Keep only requested columns that exist
    existing = [c for c in WANTED_COLUMNS if c in df_raw.columns]
    missing = [c for c in WANTED_COLUMNS if c not in df_raw.columns]
    if missing:
        print(f"These columns were not found: {missing}")

    df = df_raw[existing].copy()
    df = uppercase_strings(df)

    def count_non_empty_rows(col):
        s = col.astype("string").str.strip()
        return int(((~s.isna()) & (s != "")).sum())

    # Remove any columns that have less than the MIN_COVERAGE_RATIO percent filled in
    coverage_cutoff = math.ceil(MIN_COVERAGE_RATIO * len(df))
    min_coverage_cols = df.apply(count_non_empty_rows)
    eligible_cols = [c for c in df.columns if min_coverage_cols[c] >= coverage_cutoff]
    # Create all possible pairs of columns
    pairs = list(itertools.combinations(eligible_cols, 2))

    dropped = [c for c in df.columns if c not in eligible_cols]
    if dropped:
        print(f"[coverage] Dropping {len(dropped)} columns (< {MIN_COVERAGE_RATIO:.0%} usable):")
        for c in dropped:
            print(f"  Dropped - {c}")
        print("--------------\n")


    ROW_ID = "ROW_ID"
    df[ROW_ID] = np.arange(len(df))


    # COLLECT P VALUES---------------- ----------------
    results = []

    for c1, c2 in pairs:
        table = make_crosstab(df, c1, c2, row_id=ROW_ID)

        # Skip empty or single-level tables
        if table.shape[0] < 2 or table.shape[1] < 2:
            print(f"[skip] {c1} vs {c2}"
                f"{c1 if table.shape[0] < 2 else c2}")
            continue

        try:
            chi2, p, dof, exp = chi2_contingency(table, correction=False)
        except Exception:
            chi2 = p = dof = np.nan

        results.append({
            "c1": c1, "c2": c2, "table": table,
            "chi2": chi2, "p": p, "dof": dof,
            "expected": exp,
            "n": int(table.values.sum()),
        })

    #  Sort by most significant first
    results.sort(key=lambda r: r["p"])

    # WRITE PAGES IN SORTED ORDER--------------------------------
    with PdfPages(PDF_OUTPUT) as pdf:
        for idx, r in enumerate(results, start=1):
            c1, c2, table = r["c1"], r["c2"], r["table"]
            chi2, p, dof = r["chi2"], r["p"], r["dof"]
            # p_y, use_yates = r["p_y"], r["use_yates"]

            if p is None or math.isnan(p):
                colour = "black"
                significance_marker = " [NO]"
            else:
                if p < P_THRESH_SIG:
                    colour = "green"
                    significance_marker = " [SIG]"
                elif p < P_THRESH_MARGINAL:
                    colour = "orange"
                    significance_marker = " [MARGINAL]"
                else:
                    colour = "red"
                    significance_marker = " [NO]"

                title = (
                    f"{c1} vs {c2}{significance_marker}\n"
                    f"(chi²={chi2:.3f}, p={p:.4g})"
                )

            fig, ax = plt.subplots(figsize=(10, 10))
            table.index.name = c1
            table.columns.name = c2
            draw_heatmap(ax, table, title, colour)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            print(f"[{idx}/{len(results)}] {c1} vs {c2}")
        print(f"Saved {PDF_OUTPUT}")


    # Calculates the observed-expected for the residuals
    with PdfPages(PDF_OUTPUT_RESIDUALS) as pdf:
        for idx, r in enumerate(results, start=1):
            c1, c2, table = r["c1"], r["c2"], r["table"]
            chi2, p, dof = r["chi2"], r["p"], r["dof"]

            observed = table
            expected = pd.DataFrame(r["expected"],
                                    index=observed.index,
                                    columns=observed.columns)

            residuals = observed - expected

            if p is None or math.isnan(p):
                colour = "black"
                significance_marker = " [NO]"
            else:
                if p < P_THRESH_SIG:
                    colour = "green"
                    significance_marker = " [SIG]"
                elif p < P_THRESH_MARGINAL:
                    colour = "orange"
                    significance_marker = " [MARGINAL]"
                else:
                    colour = "red"
                    significance_marker = " [NO]"


            title = (
                    f"Residuals: {r['c1']} vs {r['c2']}{significance_marker}\n"
                    f"(chi²={chi2:.3f}, p={p:.4g})"
                )

            fig, ax = plt.subplots(figsize=(10, 10))
            draw_heatmap(ax, residuals, title, colour, residual_heatmap=True)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            print(f"[residual {idx}/{len(results)}] {r['c1']} vs {r['c2']}")
    print("Saved", PDF_OUTPUT_RESIDUALS)

if __name__ == "__main__":
    main()
import argparse
from datetime import datetime
from itertools import cycle
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib import font_manager
import json
from upsetplot import UpSet, from_memberships
import matplotlib as mpl

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.sankey import Sankey
import plotly.graph_objects as go

base_font_size = 9
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": base_font_size,     
    "axes.titlesize": base_font_size + 1,
    "axes.labelsize": base_font_size,
    "xtick.labelsize": base_font_size - 1,
    "ytick.labelsize": base_font_size - 1,
    "legend.fontsize": base_font_size - 1,   
    "mathtext.fontset": "cm",                
    "mathtext.rm": "serif",                  
})



__author__ = "Feyi Adesanya"
__copyright__ = "Copyright 2024, Sustainable Systems and Methods Lab (SSM)"
__license__ = "GPL-3.0"


warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot.plotting")
warnings.filterwarnings("ignore", category=FutureWarning, module="upsetplot.data")
data_path = "data/Data extraction sheet.xlsx"
results_path = "./output/figures"

# Get colour coding
with open("data/colour_coding.json", "r") as f:
    colour_coding = json.load(f)

class Analysis:
    observation_map = {
        1: "intentOfSoSDT", # RQ1
        2: "sosDimensions", # RQ4
        3: "trlVsContributionType", #RQ6
        4: "dtServices", #RQ3,
        5: "plot_timeline_by_category",
        6: "plot_ISO_vs_SoS_dim",
        7: "trlByDomainStacked",
        8: "bubbleByDomainAndSoTS",
        9: "bubbleByDomainAndEmergence",
        10: "sots_vs_sos_heatmap",
        11: "trlByDomainHeatmap",
    }
    
    def __init__(self):
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        self.df = self.load_data()

    def load_data(self):
        df = pd.read_excel(data_path, sheet_name="Sheet1")
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)

        df["Publication year"] = pd.to_numeric(df["Publication year"], errors="coerce")
        df["Quality score"] = pd.to_numeric(df["Quality score"], errors="coerce")

        # Make sure we don't grab data from anything where the quality score is blank or has a 0
        columns_to_check = [
            "Q1: SoS is clear",
            "Q2: DT is clear",
            "Q3: Tangible contributions",
            "Q4: Reporting clarity"
        ]
        df[columns_to_check] = df[columns_to_check].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=columns_to_check)
        df = df[~(df[columns_to_check] == 0).any(axis=1)]
        return df
  
        
# =======================
# RQ 1 
# =======================  
        
    def intentOfSoSDT(self):
        df = self.df.copy()
        services_column = "Domain (Aggregated)"
        intent_col = "Intent"

        intent_domain_counts = df.groupby([intent_col, services_column], observed=True).size().reset_index(name="Count")
        pivot_df = intent_domain_counts.pivot(index=services_column, columns=intent_col, values="Count").fillna(0)

        dt_col, sos_col = pivot_df.columns[:2].tolist()
        pivot_df["total"] = pivot_df[dt_col] + pivot_df[sos_col]
        pivot_df = pivot_df.sort_values(by="total", ascending=True)

        y_categories = pivot_df.index.tolist()
        y_pos = np.arange(len(y_categories))

        fig, ax = plt.subplots(figsize=(10, 8))

        # Bar values
        dt_values = -pivot_df[dt_col].values  # Negative for mirrored effect
        sos_values = pivot_df[sos_col].values

        ax.barh(y_pos, dt_values, color=colour_coding["red"], label=dt_col)
        ax.barh(y_pos, sos_values, color=colour_coding["blue"], label=sos_col)

        for i, (dt, sos) in enumerate(zip(dt_values, sos_values)):
            if dt != 0:
                ax.text(dt - 0.5, i, f"{-int(dt)}", va='center', ha='right', fontsize=24)
            if sos != 0:
                ax.text(sos + 0.5, i, f"{int(sos)}", va='center', ha='left', fontsize=24)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_categories, fontsize=24)
        ax.set_xlabel("Number of Studies", fontsize=24)
        ax.set_xlim(-30, 10)
        ax.set_xticks(range(-30, 15, 5))
        ax.set_xticklabels([str(abs(x)) for x in range(-30, 15, 5)], fontsize=24)
        
        ax.tick_params(axis='y', which='both', length=0, labelsize=24) 
        # ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
        # ax.axvline(0, color='black', linewidth=1)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title("Intent of SoTS Studies by Domain", fontsize=28, pad=15, x=0.25)

        plt.tight_layout()

        # Save
        self.savefig("intentOfSoSDT", upper_folder="RQ1")
        plt.close()
        


# =======================
# RQ 3 
# =======================
    def dtServices(self):
        df = self.df.copy()
        services_column = "Services (Cleaned)"

        df_services = df[[services_column]]

        combo_counts = {}
        
        for entry in df_services[services_column]:
            items = entry.split(",")
            cleaned = {str(i).strip() for i in items if i.strip()}
            sorted_combo = tuple(sorted(cleaned))
           
            if sorted_combo in combo_counts:
                combo_counts[sorted_combo] += 1
            else:
                combo_counts[sorted_combo] = 1

        upset_data = from_memberships(
        memberships=list(combo_counts.keys()),
        data=list(combo_counts.values())
        )

        plt.figure(figsize=(12, 9))
        plt.rcParams.update({
            "font.size": 23,
            "axes.titlesize": 21,
            "axes.labelsize": 22,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 20
        })

        up = UpSet(upset_data, show_counts=True, sort_by='cardinality', facecolor=colour_coding["medium_blue"]).plot()

        plt.suptitle("Combinations of DT Services Across Studies")
        up["intersections"].set_ylabel("Number of Studies")

        
        for text in plt.gcf().findobj(match=plt.Text):
            if text.get_text().isdigit():
                text.set_fontsize(23)

        self.savefig("dtServices", upper_folder="RQ3")
        plt.close()
        
# =======================
# RQ 4 
# =======================   
    def sosDimensions(self):
        df = self.df.copy()
        sos_dimensions = [col for col in df.columns if isinstance(col, str) and col.startswith("SoS:")]
        
        likert_options = ["No", "Partial", "Yes"]
        counts = {col: df[col].value_counts().reindex(likert_options, fill_value=0) for col in sos_dimensions}
        counts_df = pd.DataFrame(counts).T
        
        total_responses = df.shape[0]
        percentages = counts_df.div(total_responses) * 100
        percentages.index = [col.replace("SoS: ", "") for col in percentages.index]
        
        rename_map = {
            "Dynamic Reconfiguration": "Reconfiguration",
            "Autonomy of Constituents": "Autonomy",
            "Emergence of Behaviour": "Emergence"
        }
        percentages.rename(index=rename_map, inplace=True)
        percentages = percentages.sort_values(by="No", ascending=False)
        
        no_vals = percentages["No"]
        partial_vals = percentages["Partial"]
        yes_vals = percentages["Yes"]

        # positions of bars
        left_no      = -no_vals - (partial_vals / 2.0)
        left_partial = -partial_vals / 2.0
        left_yes     =  (partial_vals / 2.0)

        # Figure/axes
        bar_height = 0.8 # controls bar height
        # fig, ax = plt.subplots(figsize=(30, len(percentages)*1)) # controls space between bars
        fig, ax = plt.subplots(figsize=(12, 6*(0.8)))
        plt.subplots_adjust(left=0.25, right=0.95)
        y_pos = np.arange(len(percentages))
        
        ax.barh(y_pos, no_vals,      left=left_no,      height=bar_height, color=colour_coding["red"],  label="No")
        ax.barh(y_pos, partial_vals, left=left_partial, height=bar_height, color=colour_coding["grey"], label="Partial")
        ax.barh(y_pos, yes_vals,     left=left_yes,     height=bar_height, color=colour_coding["blue"], label="Yes")

        # Styling
        ax.set_facecolor("#ffffff")
        ax.set_yticks(y_pos)
        font_prop = font_manager.FontProperties(size=12)
        ax.set_yticklabels(percentages.index, fontproperties=font_prop)
        ax.tick_params(axis='y', pad=15)
        ax.set_xlabel("Percentage of Studies", fontsize=17)
        ax.set_title("SoS Dimensions", fontsize=19)
        ax.tick_params(axis='both', labelsize=17)
        ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_xlim(-100, 100)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.0f}%"))

        # Labels
        min_width = 5  # threshold for showing internal labels
        font_weight = 500
        font_size = 13
        for i in range(len(percentages.index)):
            # "No"
            if no_vals.iloc[i] > 0:
                xpos = left_no.iloc[i] + no_vals.iloc[i] / 2.0
                if no_vals.iloc[i] >= min_width:
                    ax.text(xpos, i, f"{no_vals.iloc[i]:.2f}%", va='center', ha='center',
                            color='black', fontsize=font_size, weight=font_weight)

            # "Partial"
            if partial_vals.iloc[i] >= min_width:
                ax.text(0, i, f"{partial_vals.iloc[i]:.2f}%", va='center', ha='center',
                        color='black', fontsize=font_size, weight=font_weight)
            # else: nudged overlapping "partial" label slightly downwards
            #     ax.text(0, y_pos[i] - 0.3,
            #             f"{partial_vals.iloc[i]:.2f}%", va='center', ha='center',
            #             color='black', fontsize=font_size, weight=font_weight)

            # "Yes"
            if yes_vals.iloc[i] > 0:
                xpos = left_yes.iloc[i] + yes_vals.iloc[i] / 2.0
                if yes_vals.iloc[i] >= min_width:
                    ax.text(xpos, i, f"{yes_vals.iloc[i]:.2f}%", va='center', ha='center',
                            color='black', fontsize=font_size, weight=font_weight)

        plt.tight_layout()
        self.savefig("sosDimensions", upper_folder="RQ4")

            
       
       
# =======================
# RQ 6 
# =======================
    def trlVsContributionType(self):
        df = self.df.copy()
        trl_order = [
            "Initial",
            "Proof-Of-Concept",
            "Demo Prototype",
            "Deployed Prototype",
            "Operational"
        ]
        df["TRL"] = df["TRL"].str.title()
        df["TRL"] = pd.Categorical(df["TRL"], categories=trl_order, ordered=True)

        df_unique = df.drop_duplicates(subset=["Citation Code", "TRL", "Contribution type"])
        grouped = df_unique.groupby(["TRL", "Contribution type"], observed=True)["Citation Code"].nunique().reset_index(name="Count")

        pivot_df = grouped.pivot(index="TRL", columns="Contribution type", values="Count").fillna(0).astype(int)
        pivot_df = pivot_df.reindex(trl_order)

        contribution_types = ["Conceptual", "Technical", "Case Study"]
        colors = {
            "Conceptual": colour_coding["red"],
            "Technical": colour_coding["blue"],
            "Case Study": colour_coding["grey"],
        }

        x = np.arange(len(pivot_df))
        width = 0.27

        fig, ax = plt.subplots(figsize=(12, 6))
        total_studies = df["Citation Code"].nunique()

        for i, contrib in enumerate(contribution_types):
            values = pivot_df[contrib] if contrib in pivot_df else [0]*len(pivot_df)
            bars = ax.bar(x + (i - 1) * width, values, width, label=contrib, color=colors[contrib])

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    label = f"{int(height)}"

                    offset = 5
                    ax.annotate(
                                label,
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, offset),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=18)

        ax.set_ylabel("Number of Studies", fontsize=18)
        ax.set_xlabel("Contribution Type", fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xticks(x)
        
        trl_totals = df_unique.groupby("TRL", observed=True)["Citation Code"].nunique().reindex(trl_order).fillna(0).astype(int)
        trl_percents = (trl_totals / total_studies * 100)

        x_labels = [
            f"{trl}\n{count} ({percent:.2f}%)"
            for trl, count, percent in zip(trl_order, trl_totals, trl_percents)
        ]
        ax.set_xticklabels(x_labels, rotation=15, fontsize=17)
        ax.set_title("Contribution Types by TRL Level", fontsize=20, pad=10)
        ax.set_ylim(0, pivot_df.values.max() + 10) 
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        self.savefig("trlVsContributionType", upper_folder="rq6")


# =======================
# RQ Timeline 
# =======================   
    def plot_timeline_by_category(
        self,
        category_col = "SoTS Classification",
        year_col = "Publication year",
        title = None,
        min_year = None,
        max_year = 2023,
        ):

        df = self.df.dropna(subset=[year_col, category_col]).copy()
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)

        counts = (
            df.groupby([year_col, category_col])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        # Year bounds
        y_min = counts.index.min() if min_year is None else int(min_year)
        y_max = counts.index.max() if max_year is None else int(max_year)
        years = pd.Index(range(y_min, y_max + 1), name=year_col)
        counts = counts.reindex(index=years, fill_value=0)
        
        ordered_cols = ["Directed SoTS", "Acknowledged SoTS", "Collaborative SoTS", "Virtual SoTS"]
        counts = counts.reindex(columns=ordered_cols, fill_value=0)


        # Colours
        palette = [
            colour_coding["red"],  
            colour_coding["blue"], 
            colour_coding["green"],
            colour_coding["orange"],
        ]
        
        # Plot
        fig, ax = plt.subplots(figsize=(3.45, 2.3))
        counts.plot(ax=ax, color=palette) 

        font_size_default = 10
        ax.set_xlabel("Publication Year", fontsize=font_size_default)
        ax.set_ylabel("Frequency", fontsize=font_size_default)
        ax.set_title(title or f"Studies per Year by SoTS Type", pad=5, fontsize=font_size_default)
        ax.tick_params(axis='both', labelsize=font_size_default-1)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, which="both", axis="y", linewidth=0.5, alpha=0.4)
        ax.grid(True, which="major", axis="x", linewidth=0.3, alpha=0.2)

        ax.legend(title=None, fontsize=font_size_default-2)
        # fig.tight_layout()

        self.savefig("timeline_by_sots_category")


    def plot_ISO_vs_SoS_dim(self):
        df = pd.read_excel("data/ISO_DT_map_SoS.xlsx")
        flows = df.groupby(["SoS Property", "DT Requirement"]).size().reset_index(name="value")

        labels_left = flows["SoS Property"].unique().tolist()
        labels_right = flows["DT Requirement"].unique().tolist()

        left_index = {name: i for i, name in enumerate(labels_left)}
        right_index = {name: i + len(labels_left) for i, name in enumerate(labels_right)}

        flow_sources = [left_index[row["SoS Property"]] for _, row in flows.iterrows()]
        flow_targets = [right_index[row["DT Requirement"]] for _, row in flows.iterrows()]
        flow_values  = [row["value"] for _, row in flows.iterrows()]

        # SoS node colors
        sos_colors = [
            "#4C72B0", "#55A868", "#C44E52", "#8172B2",
            "#CCB974", "#64B5CD", "#FF8C61", "#E7298A"
        ][:len(labels_left)]

        sos_link_colors = [
            "rgba(76,114,176,0.5)",
            "rgba(85,168,104,0.5)",
            "rgba(196,78,82,0.5)",
            "rgba(129,114,178,0.5)",
            "rgba(204,185,116,0.5)",
            "rgba(100,181,205,0.5)",
            "rgba(255,140,97,0.5)",
            "rgba(231,41,138,0.5)"
        ][:len(labels_left)]

        link_colors = [
            sos_link_colors[left_index[row["SoS Property"]]]
            for _, row in flows.iterrows()
        ]

        node_colors = sos_colors + ["rgba(0,0,0,0)"] * len(labels_right)


        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=25,
                thickness=20,
                line=dict(color="rgba(0,0,0,0)", width=0),
                label=labels_left + labels_right,
                color=node_colors
            ),
            link=dict(
                source=flow_sources,
                target=flow_targets,
                value=flow_values,
                color=link_colors
            )
        )])

        fig.update_layout(
            title="ISO 23247 Digital Twin Requirements Mapped to SoS Dimensions",
            title_font=dict(
                family="Times New Roman",
                size=22,
                color="black"
            ),
            title_x=0.5,  
            title_y=0.95,
            width=1400,
            height=900,
            font=dict(  
                family="Times New Roman",
                size=18,
                color="black"
            ),
            margin=dict(l=20, r=20, t=80, b=20)
        )

        fig.write_image("output/figures/overall/ISO_DT_SoS.png", scale=3)


    def trlByDomainStacked(self):
        df = self.df.copy()

        domain_col = "Domain (Aggregated)"
        trl_col = "TRL"

        trl_order = [
            "Initial",
            "Proof-Of-Concept",
            "Demo Prototype",
            "Deployed Prototype",
            "Operational"
        ]

        df[trl_col] = df[trl_col].str.title()
        df[trl_col] = pd.Categorical(df[trl_col], categories=trl_order, ordered=True)

        domain_counts = (
            df
            .groupby(domain_col)["Citation Code"]
            .nunique()
        )

        low_freq_domains = domain_counts[domain_counts < 3].index

        df.loc[
            df[domain_col].isin(low_freq_domains),
            domain_col
        ] = "Other"

        counts = (
            df
            .groupby([domain_col, trl_col], observed=True)["Citation Code"]
            .nunique()
            .unstack(fill_value=0)
            .reindex(columns=trl_order, fill_value=0)
        )

        counts["Total"] = counts.sum(axis=1)
        counts = counts.sort_values(by="Total", ascending=True)
        counts = counts.drop(columns="Total")

        # collapse small domains into other
        if "Other" in counts.index:
            other_row = counts.loc[["Other"]]
            counts = counts.drop(index="Other")
            counts = pd.concat([counts, other_row])

        # plot
        fig, ax = plt.subplots(figsize=(10, 6))

        bottom = np.zeros(len(counts))
        x = np.arange(len(counts))

        trl_colors = {
            "Initial": colour_coding["red"],
            "Proof-Of-Concept": colour_coding["orange"],
            "Demo Prototype": colour_coding["purple"],
            "Deployed Prototype": colour_coding["green"],
            "Operational": colour_coding["blue"],
        }

        base_font = 16
        title_font = base_font + 2

        for trl in trl_order:
            values = counts[trl].values

            ax.bar(
                x,
                values,
                bottom=bottom,
                label=trl,
                color=trl_colors.get(trl, "#cccccc")
            )
            for i, v in enumerate(values):
                if v > 0:
                    ax.text(
                        x[i],
                        bottom[i] + v / 2,
                        str(int(v)),
                        ha="center",
                        va="center",
                        fontsize=10
                    )

            bottom += values

        ax.set_xticks(x)
        ax.tick_params(axis='both', labelsize=base_font)
        ax.set_xticklabels(counts.index, rotation=30, ha="right")
        ax.set_ylabel("Number of Studies", fontsize=base_font)
        ax.set_xlabel("Domain", fontsize=base_font)
        ax.set_title("TRL Distribution by Domain", fontsize=title_font)

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        ax.legend(
            title="TRL Level",
            reverse=True,
            ncol=1,
            frameon=True,
            fontsize=base_font-2,
            title_fontsize=base_font-2,
            loc="best"
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        self.savefig("trlByDomainStacked", upper_folder="rq6")
        plt.close()


    def bubbleByDomainAndSoTS(self):
        df = self.df.copy()

        domain_col = "Domain (Aggregated)"
        sots_col = "SoTS Classification"
        study_col = "Citation Code"

        sots_order = [
            "Virtual SoTS",
            "Collaborative SoTS",
            "Acknowledged SoTS",
            "Directed SoTS",
        ]

        df[sots_col] = pd.Categorical(
            df[sots_col],
            categories=sots_order,
            ordered=True
        )

        domain_counts = (
            df.groupby(domain_col)[study_col]
            .nunique()
        )

        low_freq_domains = domain_counts[domain_counts < 3].index
        df.loc[df[domain_col].isin(low_freq_domains), domain_col] = "Other"

        counts = (
            df.groupby([domain_col, sots_col])[study_col]
            .nunique()
            .reset_index(name="Count")
        )

        domain_order = (
            counts.groupby(domain_col)["Count"]
            .sum()
            .sort_values(ascending=True)
            .index.tolist()
        )

        if "Other" in domain_order:
            domain_order = [d for d in domain_order if d != "Other"] + ["Other"]

        x_map = {d: i for i, d in enumerate(domain_order)}
        y_map = {s: i for i, s in enumerate(sots_order)}

        counts["x"] = counts[domain_col].map(x_map)
        counts["y"] = counts[sots_col].map(y_map)

        # bubble
        size_scale = 260
        counts["size"] = counts["Count"] * size_scale

        norm = Normalize(
            vmin=counts["Count"].min(),
            vmax=counts["Count"].max()
        )

        # plot
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.margins(x=0.1, y=0.15)

        sc = ax.scatter(
            counts["x"],
            counts["y"],
            s=counts["size"],
            c=counts["Count"],
            cmap=plt.cm.Blues,
            norm=norm,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.6
        )

        for _, row in counts.iterrows():
            if row["Count"] > 0:
                ax.text(
                    row["x"],
                    row["y"],
                    str(int(row["Count"])),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if row["Count"] >= 7 else "black",
                    weight="bold"
                )

        base_font = 15
        title_font = base_font + 3

        ax.set_xticks(range(len(domain_order)))
        ax.set_xticklabels(domain_order, rotation=30, ha="right", fontsize=base_font)

        ax.set_yticks(range(len(sots_order)))
        ax.set_yticklabels(sots_order, fontsize=base_font)

        ax.set_title(
            "Distribution of Studies by SoTS Classification and Domain",
            fontsize=title_font
        )

        ax.set_xlim(-0.5, len(domain_order) - 0.5)
        ax.set_ylim(-0.5, len(sots_order) - 0.5)
        ax.invert_yaxis()

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Number of Studies", fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        plt.tight_layout()
        self.savefig("bubbleByDomainAndSoTS", upper_folder="rq2")
        plt.close()


    def bubbleByDomainAndEmergence(self):
        df = self.df.copy()

        domain_col = "Domain (Aggregated)"
        emergence_col = "Emergence"
        study_col = "Citation Code"

        emergence_order = [
            "Not Addressed",
            "Simple",
            "Weak",
            "Strong"
        ]

        df[emergence_col] = pd.Categorical(
            df[emergence_col],
            categories=emergence_order,
            ordered=True
        )

        domain_counts = (
            df.groupby(domain_col)[study_col]
            .nunique()
        )

        low_freq_domains = domain_counts[domain_counts < 3].index
        df.loc[df[domain_col].isin(low_freq_domains), domain_col] = "Other"

        counts = (
            df.groupby([domain_col, emergence_col])[study_col]
            .nunique()
            .reset_index(name="Count")
        )

        domain_order = (
            counts.groupby(domain_col)["Count"]
            .sum()
            .sort_values(ascending=True)
            .index.tolist()
        )

        if "Other" in domain_order:
            domain_order = [d for d in domain_order if d != "Other"] + ["Other"]


        x_map = {d: i for i, d in enumerate(domain_order)}
        y_map = {e: i for i, e in enumerate(emergence_order)}

        counts["x"] = counts[domain_col].map(x_map)
        counts["y"] = counts[emergence_col].map(y_map)

        # Bubbles
        size_scale = 260
        counts["size"] = counts["Count"] * size_scale

        
        norm = Normalize(
            vmin=counts["Count"].min(),
            vmax=counts["Count"].max()
        )

        # plot
        fig, ax = plt.subplots(figsize=(11, 6))

        sc = ax.scatter(
            counts["x"],
            counts["y"],
            s=counts["size"],
            c=counts["Count"],
            cmap=plt.cm.Blues,
            norm=norm,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.6
        )

        for _, row in counts.iterrows():
            if row["Count"] > 0:
                ax.text(
                    row["x"],
                    row["y"],
                    str(int(row["Count"])),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if row["Count"] >= 7 else "black",
                    weight="bold"
                )

        base_font = 15
        title_font = base_font + 3

        ax.set_xticks(range(len(domain_order)))
        ax.set_xticklabels(domain_order, rotation=30, ha="right", fontsize=base_font)

        ax.set_yticks(range(len(emergence_order)))
        ax.set_yticklabels(emergence_order, fontsize=base_font)

        ax.set_title(
            "Distribution of Studies by Type of Emergence and Domain",
            fontsize=title_font
        )

        ax.margins(x=0.1, y=0.2)

        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Number of Studies", fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        plt.tight_layout()
        self.savefig("bubbleByDomainByEmergence", upper_folder="rq4")
        plt.close()




    def sots_vs_sos_heatmap(self):
        df = self.df.copy()

        sots_col = "SoTS Classification"
        sos_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("SoS:")]
        likert = ["No", "Partial", "Yes"]

        sots_order = [
            "Virtual SoTS",
            "Collaborative SoTS",
            "Acknowledged SoTS",
            "Directed SoTS",
        ]

        sos_rename_map = {
            "SoS: Autonomy of Constituents": "Autonomy",
            "SoS: Operational Independence": "Independence",
            "SoS: Geographic Distribution": "Distribution",
            "SoS: Evolutionary Development": "Evolution",
            "SoS: Dynamic Reconfiguration": "Reconfiguration",
            "SoS: Emergence of Behaviour": "Emergence",
            "SoS: Interdependence": "Interdependence",
            "SoS: Interoperability": "Interoperability",
        }

        sos_labels = [sos_rename_map.get(c, c.replace("SoS: ", "")) for c in sos_cols]

        heatmap_data = {}

        for sots in sots_order:
            sub = df[df[sots_col] == sots]
            total = sub["Citation Code"].nunique()

            counts = (
                pd.DataFrame({
                    col: sub[col].value_counts().reindex(likert, fill_value=0)
                    for col in sos_cols
                })
                .T
            )
            counts.index = sos_labels

            percentages = (counts.div(total) * 100).round(1)

            heatmap_data[sots] = {
                "counts": counts,
                "percentages": percentages,
                "total": total
            }

        # Plot
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(sots_order),
            figsize=(14, 5.5),
            sharey=True
        )

        cmap = plt.cm.Blues
        vmin, vmax = 0, 100
        im = None

        for ax, sots in zip(axes, sots_order):
            perc_data = heatmap_data[sots]["percentages"]
            count_data = heatmap_data[sots]["counts"]
            total = heatmap_data[sots]["total"]

            im = ax.imshow(
                perc_data.values,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(f"{sots} (n = {total})", fontsize=12, pad=6)

            ax.set_xticks(range(len(likert)))
            ax.set_xticklabels(likert, fontsize=11, rotation=20)

            ax.set_yticks(range(len(perc_data.index)))
            ax.set_yticklabels(perc_data.index, fontsize=11)
            ax.tick_params(axis="y", length=0)

            # Cell values (counts)
            for i in range(perc_data.shape[0]):
                for j in range(perc_data.shape[1]):
                    count_val = int(count_data.iloc[i, j])
                    perc_val = perc_data.iloc[i, j]

                    text_color = "white" if perc_val >= 50 else "black"

                    ax.text(
                        j, i,
                        f"{count_val}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=text_color
                    )

            for spine in ax.spines.values():
                spine.set_visible(False)

        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            fraction=0.03,
            pad=0.02
        )
        cbar.set_label("Number of Studies within SoTS type", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        fig.suptitle(
            "Studies Addressing SoS Principles Across SoTS Types",
            fontsize=14,
            y=0.98
        )

        self.savefig("sots_vs_sos_heatmap", upper_folder="RQ4")
        plt.close()



    def trlByDomainHeatmap(self):
        df = self.df.copy()

        domain_col = "Domain (Aggregated)"
        trl_col = "TRL"

        trl_order = [
            "Operational",
            "Deployed Prototype",
            "Demo Prototype",
            "Proof-Of-Concept",
            "Initial",  
        ]

        df[trl_col] = df[trl_col].str.title()
        df[trl_col] = pd.Categorical(df[trl_col], categories=trl_order, ordered=True)

        domain_counts = (
            df
            .groupby(domain_col)["Citation Code"]
            .nunique()
        )

        low_freq_domains = domain_counts[domain_counts < 3].index
        df.loc[df[domain_col].isin(low_freq_domains), domain_col] = "Other"

        counts = (
            df
            .groupby([trl_col, domain_col], observed=True)["Citation Code"]
            .nunique()
            .unstack(fill_value=0)
        )

        counts = counts.reindex(trl_order)

        domain_totals = counts.sum(axis=0).sort_values(ascending=True)
        counts = counts[domain_totals.index]

        # Keep Other at the end
        if "Other" in counts.columns:
            other_col = counts[["Other"]]
            counts = counts.drop(columns="Other")
            counts = pd.concat([counts, other_col], axis=1)

        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(
            counts.values,
            aspect="auto",
            cmap=plt.cm.Blues
        )

        ax.set_xticks(range(len(counts.columns)))
        ax.set_xticklabels(counts.columns, rotation=30, ha="right", fontsize=12)

        ax.set_yticks(range(len(trl_order)))
        ax.set_yticklabels(trl_order, fontsize=12)
        ax.tick_params(axis="y", length=0)

        ax.set_xlabel("Domain", fontsize=13)
        ax.set_ylabel("TRL Level", fontsize=13)
        ax.set_title("TRL Distribution by Domain", fontsize=14)

        # Cell values
        max_val = counts.values.max()
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                val = counts.iloc[i, j]
                text_color = "white" if val >= 0.5 * max_val else "black"
                ax.text(
                    j, i,
                    str(int(val)),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=text_color
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Number of Studies", fontsize=12)
        cbar.ax.tick_params(labelsize=11)

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        self.savefig("trlByDomainHeatmap", upper_folder="rq6")
        plt.close()

# =======================
# Saving and Running Script 
# =======================
    def savefig(self, func_name, file_type="pdf", upper_folder="overall"):
        filename = func_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
        folder_path = os.path.join(results_path, upper_folder)

        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"{filename}.{file_type}")

        # Remove any existing file with the same name
        for existing_file in os.listdir(folder_path):
            if existing_file.startswith(filename):
                os.remove(os.path.join(folder_path, existing_file))

        plt.savefig(file_path, dpi=600, bbox_inches="tight")  
        plt.close()
        
    def saveLatex(self, func_name, html_content):
        output_folder = results_path
        os.makedirs(output_folder, exist_ok=True)

        filename = func_name.replace(" ", "_").replace("-", "_") + ".tex"
        file_path = os.path.join(output_folder, filename)

        # Remove any existing file with the same name
        for existing_file in os.listdir(output_folder):
            if existing_file.startswith(filename):
                os.remove(os.path.join(output_folder, existing_file))

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)



    def run_all(self):
        print("Running all observations...\n")
        for obs_id, func_name in self.observation_map.items():
            print(f"Running observation {obs_id}: {func_name} ...")
            plt.clf()
            getattr(self, func_name)()

    def run_one(self, observation_id):
        if observation_id in self.observation_map:
            func_name = self.observation_map[observation_id]
            print(f"Running observation {observation_id}: {func_name} ...")
            plt.clf()
            getattr(self, func_name)()
        else:
            print(f"Error: Observation {observation_id} is not valid. Choose from {list(self.observation_map.keys())}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--observation", help="Observation Mapping ID", type=int, nargs="?")
    args = parser.parse_args()

    analysis = Analysis()

    if args.observation:
        analysis.run_one(args.observation)
    else:
        analysis.run_all()

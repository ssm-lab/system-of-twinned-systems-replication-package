import argparse
import os
import re
import warnings
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from upsetplot import UpSet
import plotly.express as px

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('future.no_silent_downcasting', True)
data_path = "data/Data extraction sheet.xlsx"
results_path = "./output"

class Analysis:
    observation_map = {
        # 1: "papersByCountry", #Extra Stats
        # 2: "distributionOfQualityScores", #Extra Stats
        # 3: "publicationTrendsOverTime", #Extra Stats
        1: "intentOfSoSDT", # RQ1
        2: "motivationsTable", #RQ1
        3: "topologyExtraction", #RQ2
        4: "coordinationExtraction", #RQ2
        5: "dtClassDistribution", # RQ3
        6: "levelOfIntegration", #RQ3
        # 7: "sosDimensionsHeatmap",# RQ4
        7: "sosDimensionsRadar", # RQ4
        8: "sosTypeVsEmergence", #RQ4
        9: "trlLevels", #RQ5
        10: "trlVsContributionType", #RQ5
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

    def papersByCountry(self):
        country_counts = self.df["Author countries"].dropna().str.split(", ").explode().value_counts().sort_index()

        # Ensure all values are integers
        country_counts = country_counts.round(0).astype(int)

        norm = Normalize(vmin=country_counts.min(), vmax=country_counts.max())
        colours = [sns.dark_palette("#85d4ff", as_cmap=True)(norm(v)) for v in country_counts]

        plt.figure(figsize=(12, 5))
        sns.barplot(x=country_counts.index, y=country_counts.values, hue=country_counts.index, 
                    palette=colours, ci=None, estimator=lambda x: sum(x))

        plt.xlabel("Country")
        plt.ylabel("Number of Papers")
        plt.title("Number of Papers by Country")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(range(0, int(max(country_counts)) + 1, 1))  # Forces integer y-ticks
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        self.savefig("papersByCountry")


        
        
    def distributionOfQualityScores(self):
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["Quality score"].dropna(), bins=10, kde=True, color="#85d4ff")
        plt.xlabel("Quality Score")
        plt.ylabel("Number of Papers")
        plt.title("Distribution of Quality Scores")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        self.savefig("distributionOfQualityScores")
        

    def publicationTrendsOverTime(self):
        papers_per_year = self.df["Publication year"].dropna().astype(int).value_counts().sort_index()

        norm = Normalize(vmin=papers_per_year.min(), vmax=papers_per_year.max())
        colours = [sns.dark_palette("#85d4ff", as_cmap=True)(norm(v)) for v in papers_per_year]

        plt.figure(figsize=(10, 5))
        ax = sns.barplot(x=papers_per_year.index, y=papers_per_year.values, hue = papers_per_year.index, palette=colours)

        plt.xlabel("Publication Year")
        plt.ylabel("Number of Papers")
        plt.title("Publication Trends Over Time")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        self.savefig("publicationTrendsOverTime")
      
      
# =======================
# RQ 1 
# =======================  
        
    def intentOfSoSDT(self, threshold=1):
        df = self.df.copy()

        intent_domain_counts = df.groupby(["Intent", "Domain (Aggregated)"]).size().reset_index(name="Count")
        total_count = intent_domain_counts["Count"].sum()

        intent_domain_counts["Percentage"] = (intent_domain_counts["Count"] / total_count) * 100

        threshold_mask = intent_domain_counts["Count"] <= threshold
        other_group = intent_domain_counts[threshold_mask].groupby("Intent")["Count"].sum().reset_index()
        other_group["Domain (Aggregated)"] = "Other"
        other_group["Percentage"] = (other_group["Count"] / total_count) * 100

        intent_domain_counts = intent_domain_counts[~threshold_mask]
        intent_domain_counts = pd.concat([intent_domain_counts, other_group], ignore_index=True)
        
        fig = px.treemap(
            intent_domain_counts, 
            path=["Intent", "Domain (Aggregated)"], 
            values="Count",
            title="Treemap of Intent and Domain Based on Frequency",
            color="Intent",
            color_discrete_sequence=px.colors.qualitative.Set2,
            custom_data=["Count", "Percentage"]
        )

        fig.update_traces(
            textinfo="label+text+value",
            texttemplate="<b>%{label}</b><br>%{value} (%{customdata[1]:.1f}%)",
            insidetextfont=dict(size=20),  
            outsidetextfont=dict(size=22) 
        )

        fig.update_layout(
            width=1500,
            height=1000,
            title={
                "y": 0.92, 
                "x": 0.5, 
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=24)
            },
            font=dict(size=20),
        )

        file_path = os.path.join(results_path, "RQ1/intentOfSoSDT.png")
        fig.write_image(file_path, scale=2)


        
        
    def motivationsTable(self):
        df = self.df.copy()
        
        motivation_column = "Motivation (Clustered)"
        citation_column = "Citation Code"

        if motivation_column not in df.columns:
            print(f"Error: Column '{motivation_column}' not found in dataset.")
            return

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        summary_df = df.groupby(motivation_column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: f"\\citepPS{{{', '.join(x.dropna().unique())}}}" if x.dropna().any() else "\\citepPS{placeholder}")
            # Citations=(citation_column, lambda x: ", ".join(f"\\citePS{{{cite}}}" for cite in x.dropna().unique()) if x.dropna().any() else "\\citePS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{Motivations in Studies}
            \label{tab:motivations}
            \begin{tabular}{@{}lll@{}}
            \toprule
            \multicolumn{1}{c}{\textbf{Motivation}} & \multicolumn{1}{c}{\textbf{Number of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """

            for _, row in summary_df.iterrows():
                category = row["Motivation (Clustered)"]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            \end{table*}"""

            return latex_table

        self.saveLatex("RQ1/motivations", generate_latex_table(summary_df))

# =======================
# RQ 2
# =======================
 
    def topologyExtraction(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = self.df.copy()

        topology_column = "Topology of DT/PT"
        topology_categories = ["Hierarchical", "Centralized", "Decentralized", "Distributed", "Federated"]

        def extract_topology(text):
            if pd.isna(text):
                return []
            found_categories = {word.capitalize() for word in re.findall(r"\b\w+\b", str(text))}
            return list(set(topology_categories).intersection(found_categories))

        df["Extracted Topologies"] = df[topology_column].apply(extract_topology)

        for topology in topology_categories:
            df[topology] = df["Extracted Topologies"].apply(lambda x: 1 if topology in x else 0)

        topology_counts = pd.DataFrame({
            "Topology": topology_categories, 
            "Count": [df[topology].sum() for topology in topology_categories] 
        })

        total_papers = topology_counts["Count"].sum()
        topology_counts["Percentage"] = (topology_counts["Count"] / total_papers) * 100

        topology_counts = topology_counts.sort_values(by="Percentage", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        indexes = np.arange(len(topology_counts))

        bars = ax.barh(indexes, topology_counts["Percentage"], color="#85d4ff")

        labels = [
            f"{row['Topology']} ({row['Count']}) \u2014 {row['Percentage']:.1f}%" 
            for _, row in topology_counts.iterrows()
        ]
        
        ax.set_yticks(indexes)
        ax.set_yticklabels(labels, ha="left")

        # Remove plot borders
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Remove x-axis ticks and labels
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # Adjust y-tick settings
        ax.tick_params(axis="y", direction="out", pad=-10)
        ax.yaxis.set_ticks_position("none")

        # Title as rotated y-axis label
        ax.set_ylabel("Topology", rotation=90, fontsize=12, labelpad=7)

        # Adjust font sizes
        labels = ax.get_yticklabels() + ax.get_xticklabels()
        for label in labels:
            label.set_fontsize(13)

        # Adjust figure size
        fig.set_size_inches(8, 0.33 * len(topology_counts))
        plt.tight_layout()

        self.savefig("topologyExtraction", upper_folder="RQ2")
        
        
    def coordinationExtraction(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = self.df.copy()

        categories = ["Peer-to-Peer (P2P)", "DT Orchestrated", "No Control", "System Orchestrated", "Hybrid"]

        for cat in categories:
            df[cat] = df["Coordination (Cleaned)"].apply(lambda x: 1 if cat in x else 0)

        counts = pd.DataFrame({
            "Coordination": categories, 
            "Count": [df[cat].sum() for cat in categories] 
        })

        total_papers = counts["Count"].sum()
        counts["Percentage"] = (counts["Count"] / total_papers) * 100

        counts = counts.sort_values(by="Percentage", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        indexes = np.arange(len(counts))

        bars = ax.barh(indexes, counts["Percentage"], color="#85d4ff")

        labels = [
            f"{row['Coordination']} ({row['Count']}) \u2014 {row['Percentage']:.1f}%" 
            for _, row in counts.iterrows()
        ]
        
        ax.set_yticks(indexes)
        ax.set_yticklabels(labels, ha="left")

        # Remove plot borders
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Remove x-axis ticks and labels
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # Adjust y-tick settings
        ax.tick_params(axis="y", direction="out", pad=-10)
        ax.yaxis.set_ticks_position("none")

        # Title as rotated y-axis label
        ax.set_ylabel("Coordination", rotation=90, fontsize=12, labelpad=7)

        # Adjust font sizes
        labels = ax.get_yticklabels() + ax.get_xticklabels()
        for label in labels:
            label.set_fontsize(13)

        # Adjust figure size
        fig.set_size_inches(8, 0.33 * len(counts))
        plt.tight_layout()

        self.savefig("coordinationExtraction", upper_folder="RQ2")
        



# =======================
# RQ 3 
# =======================
    def dtClassDistribution(self):
        df = self.df.copy()
        
        def classify_group(dt_class):
            if dt_class in ["Human-actuated digital twin", "Human-supervised digital twin", "Digital twin"]:
                return "Digital Twin Related"
            elif dt_class == "Digital shadow":
                return "Non Digital Twins Related"
            elif dt_class == "Digital model":
                return "Non Digital Twins Related"
            else:
                return "Other"

        df["DT Class Grouped"] = df["DT Class"].apply(classify_group)

        dt_class_counts = df.groupby(["DT Class Grouped", "DT Class"]).size().reset_index(name="Count")
        total_count = dt_class_counts["Count"].sum()

        dt_class_counts["Percentage"] = dt_class_counts["Count"] / total_count * 100

        fig = px.treemap(
            dt_class_counts,
            path=["DT Class Grouped", "DT Class"],
            values="Count",
            title="Treemap of DT Class Distribution",
            color="DT Class Grouped",
            color_discrete_sequence=px.colors.qualitative.Set2,
            custom_data=["Count", "Percentage"]
        )

        fig.update_traces(
            textinfo="label+text+value", 
            texttemplate="<b>%{label}</b><br>%{value} (%{customdata[1]:.1f}%)",
            insidetextfont=dict(size=20),  
            outsidetextfont=dict(size=22) 
        )

        fig.update_layout(
            width=1500,
            height=1000,
            title={
                "y": 0.92, 
                "x": 0.5, 
                "xanchor": "center",
                "yanchor": "top",
                "font": dict(size=24)
            },
            font=dict(size=20),
        )

        file_path = os.path.join(results_path, "RQ3/dtClassDistribution.png")
        fig.write_image(file_path, scale=2)
        
        
    def levelOfIntegration(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = self.df.copy()

        categories = ["Data Exchange", "Model Exchange", "Co-Simulation"]

        for cat in categories:
            df[cat] = df["Level of Integration (Cleaned)"].apply(lambda x: 1 if cat in x else 0)

        counts = pd.DataFrame({
            "Level of Integration": categories, 
            "Count": [df[cat].sum() for cat in categories] 
        })

        total_papers = counts["Count"].sum()
        counts["Percentage"] = (counts["Count"] / total_papers) * 100

        counts = counts.sort_values(by="Percentage", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        indexes = np.arange(len(counts))

        bars = ax.barh(indexes, counts["Percentage"], color="#85d4ff")

        labels = [
            f"{row['Level of Integration']} ({row['Count']}) \u2014 {row['Percentage']:.1f}%" 
            for _, row in counts.iterrows()
        ]
        
        ax.set_yticks(indexes)
        ax.set_yticklabels(labels, ha="left")

        # Remove plot borders
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Remove x-axis ticks and labels
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # Adjust y-tick settings
        ax.tick_params(axis="y", direction="out", pad=-10)
        ax.yaxis.set_ticks_position("none")

        # Title as rotated y-axis label
        ax.set_ylabel("Integration", rotation=90, fontsize=12, labelpad=7)

        # Adjust font sizes
        labels = ax.get_yticklabels() + ax.get_xticklabels()
        for label in labels:
            label.set_fontsize(13)

        # Adjust figure size
        fig.set_size_inches(8, 0.33 * len(counts))
        plt.tight_layout()

        self.savefig("levelOfIntegration", upper_folder="RQ3")

     
     
# =======================
# RQ 4 
# =======================   
    def sosDimensionsHeatmap(self):
        sos_dimensions = [str(col) for col in self.df.columns if isinstance(col, str) and col.startswith("SoS:")]
        
        renamed_dimensions = {col: col.replace("SoS: ", "") for col in sos_dimensions}
        mapping = {"No": 0, "Partial": 0.5, "Yes": 1}
        sos_dim = self.df[sos_dimensions].replace(mapping).astype(float)

        sos_dim = sos_dim.rename(columns=renamed_dimensions)

        plt.figure(figsize=(10, 6))
        sns.heatmap(sos_dim.corr(), annot=True, cmap="coolwarm_r", fmt=".2f", linewidths=0.5)

        plt.title("Correlation Heatmap of SoS Dimensions")
        self.savefig("sosDimensionsHeatmap", upper_folder="RQ4")
        
        
    def sosDimensionsRadar(self):
        df = self.df.copy()
    
        sos_dimensions = [str(col) for col in df.columns if isinstance(col, str) and col.startswith("SoS:")]

        mapping = {"No": 0, "Partial": 1, "Yes": 1}
        sos_dim = df[sos_dimensions].replace(mapping).astype(int)

        renamed_dimensions = {col: col.replace("SoS: ", "") for col in sos_dimensions}
        sos_dim = sos_dim.rename(columns=renamed_dimensions)

        # Compute frequencies
        freq_counts = sos_dim.sum()
        total_papers = len(df)  # Total number of papers for reference
        freq_percentage = (freq_counts / total_papers) * 100  # Convert to percentage

        # Labels and values
        labels = [f"{dim} ({percent:.1f}%)" for dim, percent in zip(freq_percentage.index, freq_percentage.values)]
        values = freq_percentage.values
        num_vars = len(labels)

        # Compute angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))  # Close the circle
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color="#85d4ff", alpha=0.3)
        ax.plot(angles, values, color="#85d4ff", linewidth=3, linestyle="solid")

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])

        # ax.set_xticklabels(labels, fontsize=14, ha="center", va="center_baseline", wrap=True, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="round,pad=0.3"), zorder=3) 
        for angle, label in zip(angles[:-1], labels):
            ax.text(angle, max(values) * 1.1, label, ha="center", va="center",
                    fontsize=12, color="black",
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle="round,pad=0.3"),
                    zorder=3)  # Ensure text is on top


        plt.title("SoS Dimensions", fontsize=20, pad=20, )
        ax.spines["polar"].set_visible(False) 

        self.savefig("sosDimensionsRadar", upper_folder="RQ4")

        
        
    def sosTypeVsEmergence(self):
        df = self.df.copy()
        df = df.rename(columns={"Type of SoS": "SoS Type", "Emergence": "Emergence"})

        df["Emergence"] = df["Emergence"].fillna("Not Considered")
        sos_vs_emergent = df.groupby(["SoS Type", "Emergence"]).size().unstack().fillna(0)

        # Sort SoS Types by total count
        sos_vs_emergent["Total"] = sos_vs_emergent.sum(axis=1)
        sos_vs_emergent = sos_vs_emergent.sort_values(by="Total", ascending=True)
        sos_vs_emergent = sos_vs_emergent.drop(columns=["Total"]) 

        sos_types = sos_vs_emergent.index.tolist()
        emergence_types = sos_vs_emergent.columns.tolist()

        colors = ["#ADEEE3", "#86DEB7", "#63B995", "#50723C"]

        # Bar width and spacing
        num_emergence = len(emergence_types)
        bar_height = 0.7 / num_emergence  # Smaller bars within each SoS Type
        y_positions = np.arange(len(sos_types))  # Main y-axis positions

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot bars for each emergence type within each SoS type
        for i, (emergence, color) in enumerate(zip(emergence_types, colors)):
            counts = sos_vs_emergent[emergence].values

            # Offset bars within each SoS Type
            bars = ax.barh(y_positions - (i - num_emergence / 2) * bar_height, counts, bar_height, label=emergence, color=color, alpha=0.9)

            # Add text labels inside bars (Aligned to Start of Bar)
            for bar, count in zip(bars, counts):
                if count > 0:  # Only label non-zero values
                    ax.text(bar.get_x() + 0.1,  # Small offset to the right
                            bar.get_y() + bar.get_height() / 2,  
                            f"{emergence} ({int(count)})", 
                            ha="left", va="center", fontsize=11, color="black")

        # Format labels
        sos_labels = [
            f"{sos} ({int(sos_vs_emergent.loc[sos].sum())})"
            for sos in sos_types
        ]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sos_labels, fontsize=13, ha="right")

        # Remove unnecessary plot borders
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_xlabel("Count", fontsize=13)
        ax.set_ylabel("SoS Type", fontsize=14, labelpad=10)
        ax.set_title("SoS Type vs. Emergent Behavior", fontsize=16, pad=15)

        # Remove x axis 
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")  # Remove X-axis label

        self.savefig("sosTypeVsEmergence", upper_folder="RQ4")
            
            
# =======================
# RQ 5 
# =======================
        
    def trlLevels(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = self.df.copy()

        # Define TRL levels in the desired order
        trl_order = ["Initial", "Proof-of-Concept", "Demo prototype", "Deployed prototype", "Operational"]
        df["TRL"] = pd.Categorical(df["TRL"], categories=trl_order, ordered=True)

        trl_counts = df["TRL"].value_counts().reindex(trl_order, fill_value=0)

        total_papers = trl_counts.sum()
        trl_percentage = (trl_counts / total_papers) * 100

        trl_df = pd.DataFrame({
            "TRL Level": trl_order,  
            "Count": trl_counts.values,
            "Percentage": trl_percentage.values
        })

        # Reverse order for correct top-to-bottom plotting
        trl_df = trl_df[::-1]

        fig, ax = plt.subplots(figsize=(8, 5))
        indexes = np.arange(len(trl_df))

        bars = ax.barh(indexes, trl_df["Percentage"], color="#85d4ff")

        # Label bars with TRL Level and Count
        labels = [
            f"{row['TRL Level']} ({row['Count']})" 
            for _, row in trl_df.iterrows()
        ]
        
        ax.set_yticks(indexes)
        ax.set_yticklabels(labels, ha="left")

        # Remove plot borders
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Remove x-axis ticks and labels
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # Adjust y-tick settings
        ax.tick_params(axis="y", direction="out", pad=-10)
        ax.yaxis.set_ticks_position("none")

        # Title as rotated y-axis label
        ax.set_ylabel("TRL Level", rotation=90, fontsize=12, labelpad=7)

        # Adjust font sizes
        labels = ax.get_yticklabels() + ax.get_xticklabels()
        for label in labels:
            label.set_fontsize(13)

        # Adjust figure size
        fig.set_size_inches(8, 0.33 * len(trl_df))
        plt.tight_layout()

        self.savefig("trlLevels", upper_folder="RQ5")

        
        
    
    
    def trlVsContributionType(self):
        df = self.df.copy()

        # Define TRL order (Ensure it's explicitly ordered)
        trl_order = ["Initial", "Proof-of-Concept", "Demo prototype", "Deployed prototype", "Operational"]
        df["TRL"] = pd.Categorical(df["TRL"], categories=trl_order, ordered=True)

        # Count occurrences of TRL and Contribution Type
        trl_contribution_counts = df.groupby(["TRL", "Contribution type"]).size().reset_index(name="Count")

        # Pivot table for visualization
        pivot_df = trl_contribution_counts.pivot(index="TRL", columns="Contribution type", values="Count").fillna(0)

        # Sort TRLs in predefined order (Ensure Initial is at the top)
        pivot_df = pivot_df.reindex(trl_order[::-1])  # Reverse order to put Initial at the top

        # Get TRL and Contribution type labels
        trl_types = pivot_df.index.tolist()
        contribution_types = pivot_df.columns.tolist()

        # Define colors (adjust as needed)
        colors = ["#ADEEE3", "#86DEB7", "#63B995", "#50723C"]

        # Bar settings
        num_contributions = len(contribution_types)
        bar_height = 0.7 / num_contributions  # Ensure sub-bars fit within each TRL category
        y_positions = np.arange(len(trl_types))  # Main y-axis positions (reversed for correct order)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Plot bars for each Contribution Type within each TRL
        for i, (contribution, color) in enumerate(zip(contribution_types, colors)):
            counts = pivot_df[contribution].values

            # Offset bars within each TRL Type
            bars = ax.barh(y_positions - (i - num_contributions / 2) * bar_height, counts, bar_height, 
                        label=contribution, color=color, alpha=0.9)

            # Add text labels inside bars (Aligned to Start of Bar)
            for bar, count in zip(bars, counts):
                if count > 0:  # Only label non-zero values
                    ax.text(bar.get_x() + 0.1,  # Small offset to the right
                            bar.get_y() + bar.get_height() / 2,  
                            f"{contribution} ({int(count)})", 
                            ha="left", va="center", fontsize=11, color="black")

        # Format labels (Ensure Initial is at the top)
        trl_labels = [
            f"{trl} ({int(pivot_df.loc[trl].sum())})"
            for trl in trl_types
        ]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(trl_labels, fontsize=13, ha="right")

        # Remove unnecessary plot borders
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Remove x-axis labels and ticks
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")  

        # Set title and labels
        ax.set_ylabel("TRL Level", fontsize=14, labelpad=10)
        ax.set_title("TRL Levels vs. Contribution Types", fontsize=16, pad=15)

        # Save the figure
        self.savefig("trlVsContributionType", upper_folder="RQ5")

                           
# =======================
# Saving and Running Script 
# =======================
    def savefig(self, func_name, file_type="pdf", upper_folder="overall"):
        filename = func_name.replace(" ", "_").replace("-", "_")
        folder_path = os.path.join(results_path, upper_folder)

        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path, f"{filename}.{file_type}")

        # Remove any existing file with the same name
        for existing_file in os.listdir(folder_path):
            if existing_file.startswith(filename):
                os.remove(os.path.join(folder_path, existing_file))

        plt.gcf().tight_layout()
        plt.savefig(file_path, dpi=900)  
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

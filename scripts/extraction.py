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

# Need to clean up and categorize - domain, coordination, topology
class Analysis:
    observation_map = {
        1: "papersByCountry", #Extra Stats
        # 2: "distributionOfQualityScores", #Extra Stats
        # 3: "publicationTrendsOverTime", #Extra Stats
        4: "intentOfSoSDT", # RQ1
        12: "motivations", #RQ1
        5: "topologyExtraction", #RQ2
        6: "dtClassDistribution", # RQ3
        # 7: "sosDimensionsHeatmap",# RQ4
        8: "sosDimensionsRadar", # RQ4
        9: "sosTypeVsEmergence", #RQ4
        10: "trlLevels", #RQ5
        11: "trlVsContributionType", #RQ5
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
        
        
    def intentOfSoSDT(self, threshold=1):
        df = self.df.copy()

        intent_domain_counts = df.groupby(["Intent [NEW]", "Domain (Aggregated)"]).size().reset_index(name="Count")
        total_count = intent_domain_counts["Count"].sum()

        intent_domain_counts["Percentage"] = (intent_domain_counts["Count"] / total_count) * 100

        threshold_mask = intent_domain_counts["Count"] <= threshold
        other_group = intent_domain_counts[threshold_mask].groupby("Intent [NEW]")["Count"].sum().reset_index()
        other_group["Domain (Aggregated)"] = "Other"
        other_group["Percentage"] = (other_group["Count"] / total_count) * 100

        intent_domain_counts = intent_domain_counts[~threshold_mask]
        intent_domain_counts = pd.concat([intent_domain_counts, other_group], ignore_index=True)
        
        fig = px.treemap(
            intent_domain_counts, 
            path=["Intent [NEW]", "Domain (Aggregated)"], 
            values="Count",
            title="Treemap of Intent and Domain Based on Frequency",
            color="Intent [NEW]",
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

        file_path = os.path.join(results_path, "intentOfSoSDT.png")
        fig.write_image(file_path, scale=2)


        
        
    def motivations(self):
        df = self.df.copy()
        
        motivation_column = "Motivation (Clustered)"
        
        if motivation_column not in df.columns:
            print(f"Error: Column '{motivation_column}' not found in dataset.")
            return

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        total_papers = len(df)


        summary_df = df.groupby(motivation_column).agg(
            Paper_Count=("Paper ID", "count")
        ).reset_index()


        summary_df["Percentage"] = (summary_df["Paper_Count"] / total_papers) * 100

        summary_df = summary_df.sort_values(by="Percentage", ascending=False)

        summary_df["Percentage"] = summary_df["Percentage"].round(2)
        
        def generate_latex_table(summary_df):
            latex_table = r"""
                \begin{table*}[h]
                    \centering
                    \caption{Motivations}
                    \begin{tabular}{|l|c|l|}
                        \hline
                        \textbf{Category} & \textbf{\# Papers} & \textbf{Papers} \\ 
                        \hline
                """
            for _, row in summary_df.iterrows():
                category = row["Motivation (Clustered)"]
                paper_count = row["Paper_Count"]
                percentage = row["Percentage"]
                
                latex_table += f"        {category} & \\textbf{{{paper_count}}} ({percentage}\\%) & \\cite{{placeholder}} \\\\\n        \\hline\n"
            
            latex_table += r"""    \end{tabular}
                    \label{tab:motivations}
                \end{table*}"""

            return latex_table
        self.saveHTML("motivations", generate_latex_table(summary_df))
        

        
    def sosDimensionsHeatmap(self):
        sos_dimensions = [str(col) for col in self.df.columns if isinstance(col, str) and col.startswith("SoS:")]
        
        renamed_dimensions = {col: col.replace("SoS: ", "") for col in sos_dimensions}
        mapping = {"No": 0, "Partial": 0.5, "Yes": 1}
        sos_dim = self.df[sos_dimensions].replace(mapping).astype(float)

        sos_dim = sos_dim.rename(columns=renamed_dimensions)

        plt.figure(figsize=(10, 6))
        sns.heatmap(sos_dim.corr(), annot=True, cmap="coolwarm_r", fmt=".2f", linewidths=0.5)

        plt.title("Correlation Heatmap of SoS Dimensions")
        self.savefig("sosDimensionsHeatmap")
        
        
    def sosDimensionsRadar(self):
        df = self.df.copy()
        
        sos_dimensions = [str(col) for col in df.columns if isinstance(col, str) and col.startswith("SoS:")]

        mapping = {"No": 0, "Partial": 0.5, "Yes": 1}
        sos_dim = df[sos_dimensions].replace(mapping).astype(float)

        renamed_dimensions = {col: col.replace("SoS: ", "") for col in sos_dimensions}
        sos_dim = sos_dim.rename(columns=renamed_dimensions)

        avg_scores = sos_dim.mean()

        labels = avg_scores.index
        values = avg_scores.values
        num_vars = len(labels)

        # Compute angles for radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color="#85d4ff", alpha=0.3)
        ax.plot(angles, values, color="#85d4ff", linewidth=3, linestyle="solid")

        # Add text labels for numerical values
        for angle, value, label in zip(angles[:-1], values[:-1], labels):
            ax.text(angle, value + 0.05, f"{value:.2f}", ha="center", fontsize=12, color="black")
            
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)

        plt.title("SoS Dimensions", fontsize=16, pad=20)
        ax.spines["polar"].set_visible(False)  # Hide outer circle

        self.savefig("sosDimensionsRadar")



    def topologyExtraction(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = self.df.copy()

        topology_column = "Topology of DT/PT [NEW]"
        topology_categories = {"Hierarchical", "Centralized", "Decentralized", "Distributed", "Federated"}

        def extract_topology(text):
            if pd.isna(text):
                return []
            found_categories = {word.capitalize() for word in re.findall(r"\b\w+\b", str(text))}
            return list(topology_categories.intersection(found_categories))

        df["Extracted Topologies"] = df[topology_column].apply(extract_topology)

        # binary matrix
        for topology in topology_categories:
            df[topology] = df["Extracted Topologies"].apply(lambda x: 1 if topology in x else 0)
            
        df.drop(columns=["Extracted Topologies", topology_column], inplace=True)

        # Convert to MultiIndex format
        df = df.set_index(list(topology_categories)).copy()

        plt.figure(figsize=(12, 7))
        upset = UpSet(df, subset_size="count", show_percentages=True)
        upset.plot()

        plt.title("Topology Combinations in Papers", fontsize=14, y=1.05)
        self.savefig("topologyExtraction")

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
            texttemplate="%{label}<br>%{value} (%{customdata[1]:.1f}%)"
        )

        fig.update_layout(
            width=1200,
            height=800,
            title={
                "y": 0.92, 
                "x": 0.5, 
                "xanchor": "center",
                "yanchor": "top"
            }
        )

        file_path = os.path.join(results_path, "dtClassDistribution.png")
        fig.write_image(file_path, scale=2)

        
        
    def sosTypeVsEmergence(self):
        df = self.df.copy()
        renamed_columns = {
            "Type of SoS [NEW]": "SoS Type",
            "Emergence [NEW]": "Emergence"
        }
        df = df.rename(columns=renamed_columns)

        df["Emergence"] = df["Emergence"].fillna("Not Considered")

        sos_vs_emergent = df.groupby(["SoS Type", "Emergence"]).size().unstack().fillna(0)

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            sos_vs_emergent, 
            annot=True, 
            cmap="Blues", 
            fmt=".0f", 
            linewidths=0.5, 
            linecolor="lightgray"
        )

        plt.xlabel("Emergent Behavior", fontsize=12)
        plt.ylabel("SoS Type", fontsize=12)
        plt.title("Heatmap of SoS Type vs. Emergent Behavior", fontsize=14)
        
        self.savefig("sosTypeVsEmergence")
        
        
    def trlLevels(self):
        df = self.df.copy()

        trl_order = ["Initial", "Proof-of-Concept", "Demo prototype", "Deployed prototype", "Operational"]
        df["TRL"] = pd.Categorical(df["TRL"], categories=trl_order, ordered=True)

        trl_counts = df["TRL"].value_counts().reindex(trl_order, fill_value=0) 

        plt.figure(figsize=(10, 6))
        plt.bar(trl_counts.index, trl_counts.values, color="#85d4ff", edgecolor="black")

        plt.xlabel("TRL Level", fontsize=12)
        plt.ylabel("Number of Papers", fontsize=12)
        plt.title("TRL Levels in Papers", fontsize=14)

        plt.grid(axis="y", linestyle="--", alpha=0.7)

        self.savefig("trlLevels")
        
        
    
    
    def trlVsContributionType(self):
        df = self.df.copy()

        trl_order = ["Initial", "Proof-of-Concept", "Demo prototype", "Deployed prototype", "Operational"]
        df["TRL"] = pd.Categorical(df["TRL"], categories=trl_order, ordered=True)

        trl_contribution_counts = df.groupby(["TRL", "Contribution type"]).size().reset_index(name="Count")

        pivot_df = trl_contribution_counts.pivot(index="TRL", columns="Contribution type", values="Count").fillna(0)

        pivot_df.plot(
            kind="bar", 
            figsize=(12, 6), 
            width=0.8, 
            colormap="Set3",
            edgecolor="black"
        )

        plt.xlabel("TRL Level", fontsize=12)
        plt.ylabel("Number of Papers", fontsize=12)
        plt.title("TRL Distribution Across Contribution Types", fontsize=14)

        capitalized_trl_order = [trl.title() for trl in trl_order]  
        plt.xticks(ticks=range(len(trl_order)), labels=capitalized_trl_order, rotation=45)

        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(title="Contribution Type", bbox_to_anchor=(1.05, 1), loc="upper left")

        self.savefig("trlVsContributionType")

                           

    def savefig(self, func_name, file_type = "pdf"):
        filename = func_name.replace(" ", "_").replace("-", "_")
        file_path = os.path.join(results_path, f"{filename}.{file_type}")

        # Remove any existing file with the same name
        for existing_file in os.listdir(results_path):
            if existing_file.startswith(filename):
                os.remove(os.path.join(results_path, existing_file))

        plt.gcf().tight_layout()
        plt.savefig(file_path, dpi=900)  
        plt.close()
        
    def saveHTML(self, func_name, html_content):
        output_folder = results_path
        os.makedirs(output_folder, exist_ok=True)

        filename = func_name.replace(" ", "_").replace("-", "_") + ".html"
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

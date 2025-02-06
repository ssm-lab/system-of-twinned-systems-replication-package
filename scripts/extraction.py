import argparse
import os
import re
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
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
        # 1: "papersByCountry", #Extra Stats
        # 2: "distributionOfQualityScores", #Extra Stats
        # 3: "publicationTrendsOverTime", #Extra Stats
        4: "intentOfSoSDT", # RQ1
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

        norm = Normalize(vmin=country_counts.min(), vmax=country_counts.max())
        colours = [sns.dark_palette("#85d4ff", as_cmap=True)(norm(v)) for v in country_counts]

        plt.figure(figsize=(12, 5))
        sns.barplot(x=country_counts.index, y=country_counts.values, hue=country_counts.index, palette=colours)

        plt.xlabel("Country")
        plt.ylabel("Number of Papers")
        plt.title("Number of Papers by Country")
        plt.xticks(rotation=45, ha="right")
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
        
        
    def intentOfSoSDT(self):
        intent_domain_counts = self.df.groupby(["Intent [NEW]", "Domain"]).size().reset_index(name="Count")
        fig = px.treemap(intent_domain_counts, 
            path=["Intent [NEW]", "Domain"], 
            values="Count",
            title="Treemap of Intent and Domain Based on Frequency",
            color="Intent [NEW]",
            color_discrete_sequence=px.colors.qualitative.Set2,)
        
        file_path = os.path.join(results_path, "intentOfSoSDT.html")
        fig.write_html(file_path)
            
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
        sos_dimensions = [str(col) for col in self.df.columns if isinstance(col, str) and col.startswith("SoS:")]
    
        # Numeric conversion
        mapping = {"No": 0, "Partial": 0.5, "Yes": 1}
        sos_dim = self.df[sos_dimensions].replace(mapping).astype(float)
        
        renamed_dimensions = {col: col.replace("SoS: ", "") for col in sos_dimensions}
        sos_dim = sos_dim.rename(columns=renamed_dimensions)

        avg_scores = sos_dim.mean()

        labels = avg_scores.index
        values = avg_scores.values
        num_vars = len(labels)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        values = np.concatenate((values, [values[0]]))
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.fill(angles, values, color="#85d4ff", alpha=0.3)
        ax.plot(angles, values, color="#85d4ff", linewidth=3, linestyle="solid")

        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=12)


        plt.title("SoS Dimensions", fontsize=16, pad=30)
        ax.spines["polar"].set_visible(False)
        self.savefig("sosDimensionsRadar")
        
    
        
    def topologyExtraction(self):
        topology_column = "Topology of DT/PT [NEW]"
        topology_categories = {"hierarchical", "centralized", "decentralized", "distributed", "federated"}

        def extract_topology(text):
            if pd.isna(text):
                return None
            found_categories = {word.lower() for word in re.findall(r"\b\w+\b", str(text))}
            return list(topology_categories.intersection(found_categories)) 

        extracted_topologies = self.df[topology_column].apply(extract_topology)
        topology_counts = extracted_topologies.explode().value_counts()

        colors = sns.color_palette("Blues", len(topology_counts))

        fig, ax = plt.subplots(figsize=(10, 6))
        squarify.plot(
            sizes=topology_counts.values, 
            label=[f"{key.capitalize()}\n({value} papers)" for key, value in topology_counts.items()], 
            alpha=0.8,
            color=colors,
            text_kwargs={'fontsize': 12, 'color': 'black'}
        )

        plt.title("DT/PT Topology Categories", fontsize=16, pad=20)
        plt.axis("off")
        self.savefig("topologyExtraction")
        
    def dtClassDistribution(self):
        dt_class_column = "DT Class" 

        dt_class_counts = self.df[dt_class_column].value_counts()
        dt_class_counts.index = dt_class_counts.index.str.title()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            y=dt_class_counts.index, 
            x=dt_class_counts.values, 
            hue=dt_class_counts.index,
            palette="Blues",
            legend=False,
        )
        plt.xlabel("Number of Papers", fontsize=12)
        plt.ylabel("DT Class", fontsize=12)
        plt.title("Distribution of DT Classes", fontsize=14)

        plt.grid(axis="x", linestyle="--", alpha=0.7)
        self.savefig("dtClassDistribution")
        
        
    def sosTypeVsEmergence(self):
        sos_column_original = "Type of SoS [NEW]"
        emergent_column_original = "Emergence [NEW]"
        renamed_columns = {
            sos_column_original: "SoS Type",
            emergent_column_original: "Emergence"
        }
        self.df = self.df.rename(columns=renamed_columns)

        sos_column = "SoS Type"
        emergent_column = "Emergence"
        
        # Replace blanks with 'Not Considered'
        self.df[emergent_column] = self.df[emergent_column].fillna("Not Considered")

        sos_vs_emergent = self.df.groupby([sos_column, emergent_column]).size().unstack()

        sos_vs_emergent.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set3")
        
        plt.xlabel("SoS Type", fontsize=12)
        plt.ylabel("Number of Papers", fontsize=12)
        plt.title("SoS Type vs. Emergent Behavior", fontsize=14)

        plt.xticks(rotation=45, ha="right")

        self.savefig("sosTypeVsEmergence")
        
        
    def trlLevels(self):
        df = self.df.copy()
        # TRL order
        trl_order = ["Initial", "Proof-of-Concept", "Demo prototype", "Deployed prototype", "Operational"]
        df["TRL"] = pd.Categorical(df["TRL"], categories=trl_order, ordered=True)
        
        trl_column = "TRL"
        trl_counts = df[trl_column].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        trl_counts.index = trl_counts.index.astype(str).str.title()
        plt.stem(trl_counts.index, trl_counts.values, linefmt="#85d4ff", markerfmt="o", basefmt=" ")

        plt.xlabel("TRL Level", fontsize=12)
        plt.ylabel("Number of Papers", fontsize=12)
        plt.title("TRL Levels in Papers", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        self.savefig("trlLevels")
        
    def trlVsContributionType(self):
        df = self.df.copy()
        trl_mapping = {
            "Initial": 1.5,  # 1-2
            "Proof-of-Concept": 3.5,  # 3-4
            "Demo prototype": 5.5,  # 5-6
            "Deployed prototype": 7.5,  # 7-8
            "Operational": 9  # 9
        }
        df["TRL_numeric"] = df["TRL"].map(trl_mapping)
    
        trl_order = ["Initial", "Proof-of-Concept", "Demo prototype", "Deployed prototype", "Operational"]
        df["TRL"] = pd.Categorical(df["TRL"], categories=trl_order, ordered=True)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Contribution type", y="TRL_numeric", data=df, palette="pastel", hue="Contribution type")
        plt.yticks(
            list(trl_mapping.values()), 
            [f"{k.title()} ({int(v)})" if isinstance(v, int) else f"{k.title()} ({v-0.5:.1f}-{v+0.5:.1f})" for k, v in trl_mapping.items()]
        )
        plt.ylabel("TRL")
        plt.title("TRL Distribution Across Contribution Types")
        plt.xticks(rotation=45)
        self.savefig("trlVsContributionType")
                           

    def savefig(self, func_name, file_type = "pdf"):
        filename = func_name.replace(" ", "_").replace("-", "_")
        file_path = os.path.join(results_path, f"{filename}.{file_type}")

        # Remove any existing file with the same name
        for existing_file in os.listdir(results_path):
            if existing_file.startswith(filename):
                os.remove(os.path.join(results_path, existing_file))

        plt.gcf().tight_layout()
        plt.savefig(file_path)
        plt.close()


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

import argparse
import os
import re
import warnings
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FuncFormatter
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec

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
        7: "sosDimensions", # RQ4
        8: "sosTypeVsEmergence", #RQ4
        9: "trlLevels", #RQ5
        10: "trlVsContributionType", #RQ5
        13: "trlTable", #RQ5
        11: "standardsTable", #RQ5
        12: "frameworksBarCharts", #RQ2
        14: "frameworksBarChartsSeperate", #RQ2
        15: "emergenceTable", #RQ4
        16: "sosTypeTable", #RQ4
        17: "dtClassTable", #RQ3
        18: "levelOfIntegrationTable", #RQ3
        19: "coordinationExtractionTable", #RQ2
        20: "topologyExtractionTable", #RQ2
        21: "EvaluationTable", #RQ5
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
        
    def intentOfSoSDT(self):
        df = self.df.copy()

       # Group by "Intent" and "Domain (Aggregated)" to get counts
        intent_domain_counts = df.groupby(["Intent", "Domain (Aggregated)"]).size().reset_index(name="Count")

        # Pivot the data so that rows are domains and columns are intents
        pivot_df = intent_domain_counts.pivot(index="Domain (Aggregated)", columns="Intent", values="Count").fillna(0)

        # Ensure we have the two expected intents. If available, use them:
        if "DT" in pivot_df.columns and "SoS" in pivot_df.columns:
            dt_col = "DT"
            sos_col = "SoS"
        else:
            # Fallback: use the first column as DT and the second as SoS
            dt_col, sos_col = pivot_df.columns[0], pivot_df.columns[1]

        # Create ordering groups using the original (non-mirrored) counts:
        # Group 1: SoS-exclusive (DT==0 and SoS>0)
        # Group 2: Common (DT>0 and SoS>0)
        # Group 3: DT-exclusive (DT>0 and SoS==0)
        pivot_df["order_group"] = 2  # default: common
        pivot_df.loc[(pivot_df[dt_col] == 0) & (pivot_df[sos_col] > 0), "order_group"] = 1
        pivot_df.loc[(pivot_df[dt_col] > 0) & (pivot_df[sos_col] == 0), "order_group"] = 3

        # Compute the difference for further ordering (SoS minus DT)
        pivot_df["difference"] = pivot_df[sos_col] - pivot_df[dt_col]

        # Now sort first by the defined group and then by the difference (descending)
        pivot_df = pivot_df.sort_values(by=["order_group", "difference"], ascending=[True, False])
        y_categories = pivot_df.index.tolist()

        # Mirror the DT column for plotting (so that DT bars extend to the left)
        pivot_df["DT_mirrored"] = -pivot_df[dt_col]

        # Create the mirror (population pyramid–style) horizontal bar chart using Plotly
        fig = go.Figure()

        # Trace for DT (mirrored to the left)
        fig.add_trace(go.Bar(
            y=y_categories,
            x=[pivot_df.loc[domain, "DT_mirrored"] for domain in y_categories],
            name=dt_col,
            orientation='h',
            marker_color='rgb(222,45,38)',
            text=[f"{pivot_df.loc[domain, dt_col]:.0f}" for domain in y_categories],
            textposition='inside',
            hovertemplate='Domain: %{y}<br>' + dt_col + ': %{text}<extra></extra>',
        ))

        # Trace for SoS (to the right)
        fig.add_trace(go.Bar(
            y=y_categories,
            x=[pivot_df.loc[domain, sos_col] for domain in y_categories],
            name=sos_col,
            orientation='h',
            marker_color='rgb(49,130,189)',
            text=[f"{pivot_df.loc[domain, sos_col]:.0f}" for domain in y_categories],
            textposition='inside',
            hovertemplate='Domain: %{y}<br>' + sos_col + ': %{text}<extra></extra>',
        ))

        # Determine a common maximum for symmetry
        max_count = max(pivot_df[dt_col].max(), pivot_df[sos_col].max())

        fig.update_layout(
            title="Intent by Domain",
            barmode='overlay',
            bargap=0.1,
            xaxis=dict(
                tickvals=[-max_count, -max_count/2, 0, max_count/2, max_count],
                ticktext=[str(int(max_count)), str(int(max_count/2)), "0", str(int(max_count/2)), str(int(max_count))],
                title="# of Studies",
                showgrid=True,
            ),
            yaxis=dict(
                title="Domain",
                automargin=True,
                showgrid=True,
            ),
            width=1400,
            height=900,
            font=dict(size=18),
            plot_bgcolor="white",
            legend=dict(
                x=0.65, 
                y=1,
                font=dict(size=18), 
                bgcolor="white", 
            )
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
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{Motivations in Studies}
            \label{tab:motivations}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{5cm} l p{12.5cm} @{}}
            \toprule
            \multicolumn{1}{c}{\textbf{TRL}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                category = row["Motivation (Clustered)"]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ1/motivations", generate_latex_table(summary_df))


# =======================
# RQ 2
# =======================
 
    def topologyExtraction(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = self.df.copy()
        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

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

        total_papers = df["Paper ID"].nunique()  # Get total unique papers
        topology_counts["Percentage"] = (topology_counts["Count"] / total_papers) * 100

        topology_counts = topology_counts.sort_values(by="Percentage", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, 100)
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
        

    def topologyExtractionTable(self):
        df = self.df.copy()
        
        topology_column = "Topology of DT/PT"
        citation_column = "Citation Code"
        topology_categories = ["Hierarchical", "Centralized", "Decentralized", "Distributed", "Federated"]
        
        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]
        
        def extract_topology(text):
            if pd.isna(text):
                return []
            found_categories = {word.capitalize() for word in re.findall(r"\b\w+\b", str(text))}
            return list(set(topology_categories).intersection(found_categories))

        df["Extracted Topologies"] = df[topology_column].apply(extract_topology)
        
        exploded_df = df.explode("Extracted Topologies")
        
        summary_df = exploded_df.groupby("Extracted Topologies").agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                        if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()
        
        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)
        
        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{Topologies in Studies}
            \label{tab:rq2-topology}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{1.5cm} l p{17cm} @{} }
            \toprule
            \multicolumn{1}{c}{\textbf{Topology}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                topology = row["Extracted Topologies"]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                
                # Wrap citations in \texttt{} to avoid formatting issues
                wrapped_citations = " ".join(citations.split(", "))
                
                latex_table += f"{topology} & \\maindatabar{{{paper_count}}} & \\texttt{{{wrapped_citations}}} \\\\ \n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table
        
        self.saveLatex("RQ2/topologyExtractionTable", generate_latex_table(summary_df))


        
        
    def coordinationExtraction(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = self.df.copy()
        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        categories = ["Peer-to-Peer (P2P)", "DT Orchestrated", "No Control", "System Orchestrated", "Hybrid"]

        for cat in categories:
            df[cat] = df["Coordination (Cleaned)"].apply(lambda x: 1 if cat in x else 0)

        counts = pd.DataFrame({
            "Coordination": categories, 
            "Count": [df[cat].sum() for cat in categories] 
        })

        total_papers = df["Paper ID"].nunique()  # Get total unique papers
        counts["Percentage"] = (counts["Count"] / total_papers) * 100

        counts = counts.sort_values(by="Percentage", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, 100)
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
        
    def coordinationExtractionTable(self):
        df = self.df.copy()
        
        column = "Coordination (Cleaned)"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{Coordination in Studies}
            \label{tab:rq2-coordination}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{3.5cm} l p{15cm} @{}}
            \toprule
            \multicolumn{1}{c}{\textbf{Coordination}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                category = row[column]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ2/coordinationExtractionTable", generate_latex_table(summary_df))
        
        
        
    def frameworksBarCharts(self, threshold=1):
        df = self.df.copy()
        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]
        
        framework_columns = {
            "Digital Twin & IoT": "DT/IoT", 
            "Modeling & Simulation": "Modeling/Sim", 
            "AI, Data Analytics & Machine Learning": "Analytics", 
            "Cloud, Edge, and DevOps": "CloudOps", 
            "Systems Engineering & Architecture": "Sys Eng & Arch", 
            "Data Management": "Data Mgmt", 
            "Geospatial & Visualization Technologies": "Geo/Viz", 
            "Application Development & Web Technologies": "App/Web Dev"
        }
        
        data_list = []
        
        for full_col, short_label in framework_columns.items():
            # Extract individual frameworks.
            rows = []
            for _, row in df.iterrows():
                cell = row[full_col]
                if pd.isna(cell):
                    continue
                frameworks_list = [fw.strip() for fw in str(cell).split(",") if fw.strip()]
                for fw in frameworks_list:
                    rows.append({
                        "Framework": fw,
                        "Paper ID": row["Paper ID"]
                    })
            
            if not rows:
                continue
            
            exploded_df = pd.DataFrame(rows)
            
            # Group by framework to count the number of unique papers.
            summary_df = exploded_df.groupby("Framework").agg(
                Paper_Count=("Paper ID", "nunique")
            ).reset_index()
            
            # Group frameworks that appear in <= threshold papers into "Other"
            mask = summary_df["Paper_Count"] <= threshold
            other_count = summary_df.loc[mask, "Paper_Count"].sum()
            other_row = None

            # If all categories would be grouped as "Other", don't group
            if mask.sum() == len(summary_df):  # Everything falls into "Other"
                pass  # Leave summary_df as is
            else:
                if mask.sum() > 0:
                    other_row = pd.DataFrame([{"Framework": "Other", "Paper_Count": other_count}])
                    summary_df = summary_df[~mask]

            # Sort the remaining frameworks by Paper_Count (ascending)
            summary_df = summary_df.sort_values(by="Paper_Count", ascending=True)

            # If "Other" exists, add it at the bottom
            if other_row is not None:
                summary_df = pd.concat([summary_df, other_row], ignore_index=True)

            # Ensure "Other" always appears at the bottom
            if "Other" in summary_df["Framework"].values:
                summary_df = summary_df.sort_values(by=["Framework"], key=lambda x: x != "Other")

            
            # Calculate percentages.
            total_papers = df["Paper ID"].nunique()  # Get total unique papers
            summary_df["Percentage"] = (summary_df["Paper_Count"] / total_papers) * 100
            
            # Save the processed data.
            data_list.append((full_col, short_label, summary_df, len(summary_df)))
        
        # If no valid data, exit.
        if not data_list:
            print("No data found for any framework categories.")
            return
        
        # Compute total height for the figure.
        total_relative_height = sum([n for (_, _, _, n) in data_list])
        total_height = 0.33 * total_relative_height + 2  # Add extra margin
        
        # Create a figure with subplots arranged vertically.
        n_subplots = len(data_list)
        height_ratios = [n for (_, _, _, n) in data_list]
        
        fig = plt.figure(figsize=(8,  max(total_height, 4)))
        gs = gridspec.GridSpec(n_subplots, 1, height_ratios=height_ratios, hspace=0.6)
        
        # Loop through each category and plot in its respective subplot.
        for i, (full_col, short_label, summary_df, n_rows) in enumerate(data_list):
            ax = fig.add_subplot(gs[i])
            indexes = np.arange(len(summary_df))
            ax.barh(indexes, summary_df["Percentage"], color="#85d4ff")
            ax.set_xlim(0, 100)
            
            # Create labels like: "Framework (Count) — Percentage%"
            labels = [
                f"{row['Framework']} ({row['Paper_Count']}) — {row['Percentage']:.1f}%"
                for _, row in summary_df.iterrows()
            ]
            ax.set_yticks(indexes)
            ax.set_yticklabels(labels, ha="left")
            
            # Remove plot borders and x-axis ticks/labels.
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis="y", direction="out", pad=-10)
            ax.yaxis.set_ticks_position("none")
            
            # Use the short label as the rotated y-axis title.
            ax.set_ylabel(short_label, rotation=90, fontsize=12, labelpad=7)
            
            # Adjust font sizes.
            for label in ax.get_yticklabels() + ax.get_xticklabels():
                label.set_fontsize(13)
        
        plt.tight_layout()
        self.savefig("frameworksBarCharts", upper_folder="RQ2")
        
        
    def frameworksBarChartsSeperate(self, threshold=1):
        df = self.df.copy()
        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        framework_columns = {
            "Digital Twin & IoT": "DT/IoT", 
            "Modeling & Simulation": "Modeling/Sim", 
            "AI, Data Analytics & Machine Learning": "Analytics", 
            "Cloud, Edge, and DevOps": "CloudOps", 
            "Systems Engineering & Architecture": "Sys Eng & Arch", 
            "Data Management": "Data Mgmt", 
            "Geospatial & Visualization Technologies": "Geo/Viz", 
            "Application Development & Web Technologies": "App/Web Dev"
        }

        for full_col, short_label in framework_columns.items():
            rows = []
            for _, row in df.iterrows():
                cell = row[full_col]
                if pd.isna(cell):
                    continue
                frameworks_list = [fw.strip() for fw in str(cell).split(",") if fw.strip()]
                for fw in frameworks_list:
                    rows.append({
                        "Framework": fw,
                        "Paper ID": row["Paper ID"]
                    })

            if not rows:
                continue  # Skip empty categories

            exploded_df = pd.DataFrame(rows)

            # Count occurrences per framework
            summary_df = exploded_df.groupby("Framework").agg(
                Paper_Count=("Paper ID", "nunique")
            ).reset_index()

            # Group low-frequency frameworks into "Other"
            mask = summary_df["Paper_Count"] <= threshold
            other_count = summary_df.loc[mask, "Paper_Count"].sum()

            if mask.sum() == len(summary_df):  # If everything falls into "Other", keep all instead
                pass
            else:
                if mask.sum() > 0:
                    other_row = pd.DataFrame([{"Framework": "Other", "Paper_Count": other_count}])
                    summary_df = summary_df[~mask]
                    summary_df = pd.concat([summary_df, other_row], ignore_index=True)

            # Sort frameworks (ascending), ensuring "Other" is always last
            summary_df = summary_df.sort_values(by="Paper_Count", ascending=True)
            if "Other" in summary_df["Framework"].values:
                summary_df = summary_df.sort_values(by=["Framework"], key=lambda x: x != "Other")

            # Calculate percentages based on total unique papers
            total_papers = df["Paper ID"].nunique()
            summary_df["Percentage"] = (summary_df["Paper_Count"] / total_papers) * 100


            indexes = np.arange(len(summary_df))
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, 0.33 * max(len(summary_df), 4))) # no cut off for y axis title
            ax.barh(indexes, summary_df["Percentage"], color="#85d4ff")
            ax.set_xlim(0, 100)  # Ensure consistent scaling

            # Labels like: "Framework (Count) — Percentage%"
            labels = [
                f"{row['Framework']} ({row['Paper_Count']}) — {row['Percentage']:.1f}%"
                for _, row in summary_df.iterrows()
            ]
            ax.set_yticks(indexes)
            ax.set_yticklabels(labels, ha="left")

            # Formatting
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
            ax.tick_params(axis="y", direction="out", pad=-10)
            ax.yaxis.set_ticks_position("none")
            ax.set_ylabel(short_label, rotation=90, fontsize=12, labelpad=7)

            # Adjust font sizes
            for label in ax.get_yticklabels() + ax.get_xticklabels():
                label.set_fontsize(13)

            self.savefig(f"frameworksBarCharts_{short_label}", upper_folder="RQ2")
            plt.close(fig)


# =======================
# RQ 3 
# =======================
    def dtClassDistribution(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        df = self.df.copy()
        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]
        
        # Define the DT related classes
        dt_related = ["Human-actuated digital twin", "Human-supervised digital twin", "Digital twin"]
        
        # Group by DT Class only
        dt_class_counts = df.groupby("DT Class").size().reset_index(name="Count")
        total_count = df["Paper ID"].nunique()  # Get total unique papers
        dt_class_counts["Percentage"] = dt_class_counts["Count"] / total_count * 100
        
        # Mark DT related classes for ordering
        dt_class_counts["Is_DT_Related"] = dt_class_counts["DT Class"].apply(
            lambda x: 0 if x in dt_related else 1
        )
        
        # Sort so that DT related classes (Is_DT_Related=1) appear at the top.
        # Within each group, sort by percentage in ascending order.
        dt_class_counts = dt_class_counts.sort_values(
            by=["Is_DT_Related", "Percentage"], ascending=[False, True]
        )
        
        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 0.33 * len(dt_class_counts)))
        ax.set_xlim(0, 100)
        indexes = np.arange(len(dt_class_counts))
        bars = ax.barh(indexes, dt_class_counts["Percentage"], color="#85d4ff")
        
        # Format the y-axis labels: "DT Class (Count) — Percentage%"
        labels = [
            f"{row['DT Class']} ({row['Count']}) \u2014 {row['Percentage']:.1f}%"
            for _, row in dt_class_counts.iterrows()
        ]
        
        ax.set_yticks(indexes)
        ax.set_yticklabels(labels, ha="left")
        
        # Remove unnecessary plot borders and ticks for a cleaner look
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis="y", direction="out", pad=-10)
        ax.yaxis.set_ticks_position("none")
        
        # Set the y-axis label (rotated)
        ax.set_ylabel("DT Class", rotation=90, fontsize=12, labelpad=7)
        
        # Adjust tick label font sizes
        for label in ax.get_yticklabels() + ax.get_xticklabels():
            label.set_fontsize(13)
        
        plt.tight_layout()
        
        # Save the figure using the class's savefig method
        self.savefig("dtClassDistribution", upper_folder="RQ3")
        
    def dtClassTable(self):
        df = self.df.copy()
        
        column = "DT Class"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{DT Levels of Autonomy in Studies}
            \label{tab:rq3-dt-class}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{3.5cm} l p{15cm} @{}}
            \toprule
            \multicolumn{1}{c}{\textbf{Autonomy}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                category = row[column]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ3/dtClassTable", generate_latex_table(summary_df))
        

            
        
    def levelOfIntegration(self):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df = self.df.copy()
        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]


        categories = ["Data Exchange", "Model Exchange", "Co-Simulation"]

        for cat in categories:
            df[cat] = df["Level of Integration (Cleaned)"].apply(lambda x: 1 if cat in x else 0)

        counts = pd.DataFrame({
            "Level of Integration": categories, 
            "Count": [df[cat].sum() for cat in categories] 
        })

        
        total_papers = df["Paper ID"].nunique()  # Get total unique papers
        counts["Percentage"] = (counts["Count"] / total_papers) * 100

        counts = counts.sort_values(by="Percentage", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, 100)
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
        
        
    def levelOfIntegrationTable(self):
        df = self.df.copy()
        
        column = "Level of Integration (Cleaned)"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{DT Level of Integration of Constituents in Studies}
            \label{tab:rq3-lvl-integration}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{3.5cm} l p{15cm} @{}}
            \toprule
            \multicolumn{1}{c}{\textbf{Integration}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                category = row[column]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ3/levelOfIntegrationTable", generate_latex_table(summary_df))

     
     
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
        
        
    def sosDimensions(self):
        df = self.df.copy()
        sos_dimensions = [col for col in df.columns if isinstance(col, str) and col.startswith("SoS:")]
        
        likert_options = ["No", "Partial", "Yes"]
        
        counts = {}
        for col in sos_dimensions:
            counts[col] = df[col].value_counts().reindex(likert_options, fill_value=0)
        counts_df = pd.DataFrame(counts).T  # rows: dimensions, columns: responses
        
        total_responses = df.shape[0]
        percentages = counts_df.div(total_responses) * 100
        percentages.index = [col.replace("SoS: ", "") for col in percentages.index]

        no_vals = percentages["No"]
        partial_vals = percentages["Partial"]
        yes_vals = percentages["Yes"]

        # Compute left starting positions:
        left_no = -no_vals - (partial_vals / 2)
        left_partial = -partial_vals / 2
        left_yes = partial_vals / 2

        # Increase overall figure size further
        fig, ax = plt.subplots(figsize=(30, len(percentages) * 1.5))
        y_pos = np.arange(len(percentages))
        
        ax.barh(y_pos, no_vals, left=left_no, color="#3182bd", label="No")
        ax.barh(y_pos, partial_vals, left=left_partial, color="#b5bbc3", label="Partial")
        ax.barh(y_pos, yes_vals, left=left_yes, color="#de2d26", label="Yes")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(percentages.index, fontsize=16)
        ax.set_xlabel("Percentage", fontsize=18)
        ax.set_title("SoS Dimensions",  fontsize=25)
        
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=25)

        # Add a vertical line at x=0 and legend
        ax.axvline(0, color='black', linewidth=0.2)

        # Increase x-axis limits to provide more space for labels
        left_lim = left_no.min() - 2
        right_lim = (left_yes + yes_vals).max() + 2
        ax.set_xlim(left_lim, right_lim)

        # Use a MultipleLocator for larger spacing between x ticks (every 10 units)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.0f}"))
        
        # Define a threshold below which labels will be repositioned (in percentage points)
        min_width = 5

        # Add text labels within or near each bar segment with additional offsets for tiny segments
        for i, dim in enumerate(percentages.index):
            # "No" category: if tiny, position to the left of the bar
            if no_vals[i] > 0:
                if no_vals[i] < min_width:
                    ax.text(left_no[i] - 1, i, f"{no_vals[i]:.2f}%", va='center', ha='right',
                            color='white', fontsize=18, fontweight="bold")
                else:
                    ax.text(left_no[i] + no_vals[i] / 2, i, f"{no_vals[i]:.2f}%",
                            va='center', ha='center', color='white', fontsize=18, fontweight="bold")
            # "Partial" category: always centered at x=0; if tiny, nudge vertically
            if partial_vals[i] > 0:
                if partial_vals[i] < min_width:
                    ax.text(0, i - 0.3, f"{partial_vals[i]:.2f}%", va='bottom', ha='center',
                            color='white', fontsize=18, fontweight="bold")
                else:
                    ax.text(0, i, f"{partial_vals[i]:.2f}%", va='center', ha='center',
                            color='white', fontsize=18, fontweight="bold")
            # "Yes" category: if tiny, position to the right of the bar
            if yes_vals[i] > 0:
                if yes_vals[i] < min_width:
                    ax.text(left_yes[i] + yes_vals[i] / 2 + 1, i, f"{yes_vals[i]:.2f}%",
                            va='center', ha='left', color='white', fontsize=18, fontweight="bold")
                else:
                    ax.text(left_yes[i] + yes_vals[i] / 2, i, f"{yes_vals[i]:.2f}%",
                            va='center', ha='center', color='white', fontsize=18, fontweight="bold")

        self.savefig("sosDimensions", upper_folder="RQ4")

        
        
    def sosTypeTable(self):
        df = self.df.copy()
        
        column = "Type of SoS"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{SoS Type in Studies}
            \label{tab:sos-type}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{2.5cm} l p{14cm} @{}}
            \toprule
            \multicolumn{1}{c}{\textbf{TRL}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                category = row[column]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ4/sosTypeTable", generate_latex_table(summary_df))
        
        
    def emergenceTable(self):
        df = self.df.copy()
        
        column = "Emergence"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{Emergence Type in Studies}
            \label{tab:emergence-type}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{2.5cm} l p{14cm} @{}}
            \toprule
            \multicolumn{1}{c}{\textbf{TRL}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                category = row[column]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ4/emergenceTable", generate_latex_table(summary_df))
        
        
    def sosTypeVsEmergence(self):
        df = self.df.copy()
        df = df.rename(columns={"Type of SoS": "SoS Type", "Emergence": "Emergence"})
        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]


        sos_vs_emergent = df.groupby(["SoS Type", "Emergence"]).size().unstack().fillna(0)

        # Sort SoS Types by total count
        sos_vs_emergent["Total"] = sos_vs_emergent.sum(axis=1)
        sos_vs_emergent = sos_vs_emergent.sort_values(by="Total", ascending=True)
        sos_vs_emergent = sos_vs_emergent.drop(columns=["Total"]) 

        sos_types = sos_vs_emergent.index.tolist()
        emergence_types = sos_vs_emergent.columns.tolist()

        colors = ["#FFD166", "#83bd63", "#ff9a26", "#ff4646"]

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
                            ha="left", va="center", fontsize=18, color="black")

        # Format labels
        sos_labels = [
            f"{sos} ({int(sos_vs_emergent.loc[sos].sum())})"
            for sos in sos_types
        ]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(sos_labels, fontsize=18, ha="right")

        # Remove unnecessary plot borders
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        ax.set_xlabel("Count", fontsize=18)
        ax.set_ylabel("SoS Type", fontsize=18, labelpad=10)
        ax.set_title("SoS Type vs. Emergent Behavior", fontsize=16, pad=15)
        ax.legend(fontsize=16)

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

        colors = ["#3182bd", "#b5bbc3", "#de2d26"]

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
                            ha="left", va="center", fontsize=18, color="black")

        # Format labels (Ensure Initial is at the top)
        trl_labels = [
            f"{trl} ({int(pivot_df.loc[trl].sum())})"
            for trl in trl_types
        ]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(trl_labels, fontsize=18, ha="right")

        # Remove unnecessary plot borders
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        # Remove x-axis labels and ticks
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")  

        # Set title and labels
        ax.set_ylabel("TRL Level", fontsize=18, labelpad=10)
        ax.set_title("TRL Levels vs. Contribution Types", fontsize=20, pad=15)
        ax.legend(fontsize=16)

        # Save the figure
        self.savefig("trlVsContributionType", upper_folder="RQ5")
        
    def standardsTable(self, threshold = 2):
        # Use a copy of the dataframe
        df = self.df.copy()
    
        standards_column = "Standards Used (Cleaned Up)"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]
        
        # Helper function to format citations:
        # Remove nulls and duplicates, then wrap each citation individually with \citepPS{...}.
        def format_citations(citations):
            citations = [cite for cite in citations if pd.notna(cite)]
            seen = set()
            unique = []
            for c in citations:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            return ", ".join([f"\\citepPS{{{c}}}" for c in unique]) if unique else "\\citepPS{placeholder}"
        
        # Extract individual standards.
        # Assume each cell is a semicolon-separated list.
        rows = []
        for _, row in df.iterrows():
            standards = row[standards_column]
            if pd.isna(standards):
                continue
            # Split by semicolon and remove extra whitespace.
            standards_list = [s.strip() for s in str(standards).split(";") if s.strip()]
            for std in standards_list:
                rows.append({
                    "Standard": std,
                    "Paper ID": row["Paper ID"],
                    "Citation Code": row[citation_column] if citation_column in row else None
                })
        
        # Create an exploded DataFrame where each row corresponds to one standard occurrence.
        exploded_df = pd.DataFrame(rows)
        
        # Group by Standard to count unique papers and aggregate citations.
        summary_df = exploded_df.groupby("Standard").agg(
            Paper_Count=("Paper ID", "nunique"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" 
                                                            for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()
        
        # Identify low-frequency standards (those mentioned in only one paper).
        mask = summary_df["Paper_Count"] <= threshold
        other_row = pd.DataFrame()
        if mask.sum() > 0:
            # Get the list of low-frequency standards.
            low_freq_standards = summary_df[mask]["Standard"].tolist()
            other_raw_citations = exploded_df.loc[
                exploded_df["Standard"].isin(low_freq_standards), citation_column
            ].dropna().unique()
            other_row = pd.DataFrame([{
                "Standard": "Other",
                "Paper_Count": len(other_raw_citations), # count based on the number of papers, rather than number of standards (one paper might use multiple standards)
                "Citations": format_citations(other_raw_citations)
            }])
            # Remove low-frequency standards from the summary.
            summary_df = summary_df[~mask]
        
        # Sort the remaining standards by count
        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)
        
        # Append the "Other" category row at the end
        if not other_row.empty:
            summary_df = pd.concat([summary_df, other_row], ignore_index=True)
        
        # Generate a LaTeX table.
        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{Standards Used in Papers}
            \label{tab:standards}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{5cm} l p{11.5cm} @{}}
            \toprule
            \multicolumn{1}{c}{\textbf{Standards}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                standard = row["Standard"]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{standard} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"
            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ5/standards", generate_latex_table(summary_df))
        
        
    def trlTable(self):
        df = self.df.copy()
        
        column = "TRL"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{TRL in Studies}
            \label{tab:trl}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{2.5cm} l p{14cm} @{}}
            \toprule
             \multicolumn{1}{c}{\textbf{TRL}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                category = row[column]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ5/trlTable", generate_latex_table(summary_df))
        
        
    def EvaluationTable(self):
        df = self.df.copy()
        
        column = "Evaluation"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]

        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        def generate_latex_table(summary_df):
            latex_table = r"""\begin{table*}[]
            \centering
            \caption{Evaluation in Studies}
            \label{tab:rq5-evaluation}
            \resizebox{\textwidth}{!}{ 
            \begin{tabular}{@{} p{2.5cm} l p{14cm} @{}}
            \toprule
             \multicolumn{1}{c}{\textbf{Evaluation}} & \multicolumn{1}{c}{\textbf{\# of studies}} & \multicolumn{1}{c}{\textbf{Studies}} \\ 
            \midrule
            """
            for _, row in summary_df.iterrows():
                category = row[column]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += r"""\bottomrule
            \end{tabular}
            }
            \end{table*}"""
            return latex_table

        self.saveLatex("RQ5/EvaluationTable", generate_latex_table(summary_df))

                
                           
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

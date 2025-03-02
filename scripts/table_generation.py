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
results_path = "./output/tables"

class Analysis:
    observation_map = {
        1: "motivationsTable", #RQ1
        2: "topologyExtractionTable", #RQ2
        3: "coordinationExtractionTable", #RQ2
        4: "autonomyTable", #RQ3
        5: "levelOfIntegrationTable", #RQ3
        6: "emergenceTable", #RQ4
        7: "sosTypeTable", #RQ4
        8: "trlTable", #RQ5
        9: "EvaluationTable", #RQ5
        10: "standardsTable", #RQ5
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
# Table Generator
# =======================    
    def generate_latex_table(self, summary_df, caption, label, tabular_size, first_column_name):
            latex_table = f"""\\begin{{table*}}[]
            \\centering
            \\caption{{{caption}}}
            \\label{{tab:{label}}}
            \\resizebox{{\\textwidth}}{{!}}{{ 
            \\begin{{tabular}}{{@{{}} {tabular_size} @{{}}}}
            \\toprule
            \\multicolumn{{1}}{{c}}{{\\textbf{{{first_column_name}}}}} & 
            \\multicolumn{{1}}{{c}}{{\\textbf{{\\# of studies}}}} & 
            \\multicolumn{{1}}{{c}}{{\\textbf{{Studies}}}} \\\\ 
            \\midrule
            """
            for _, row in summary_df.iterrows():
                category = row.iloc[0]
                paper_count = row["Paper_Count"]
                citations = row["Citations"]
                latex_table += f"{category} & \\maindatabar{{{paper_count}}} & {citations} \\\\\n"

            latex_table += """\\bottomrule
            \\end{tabular}
            }
            \\end{table*}"""
            return latex_table

      
# =======================
# RQ 1 
# =======================          
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
        
        caption = "Motivations in Studies"
        label = "motivations"
        tabular_size = "p{5cm} l p{12.5cm}"
        first_column_name = "Motivation"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)

        self.saveLatex("RQ1/motivations", latex_table)


# =======================
# RQ 2
# =======================

        

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
        
        caption = "Topologies in Studies"
        label = "rq2-topology"
        tabular_size = "p{3.5cm} l p{15cm}"
        first_column_name = "Topology"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ2/topologyExtractionTable", latex_table)


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
        
        caption = "Coordination in Studies"
        label = "rq2-coordination"
        tabular_size = "p{3.5cm} l p{15cm}"
        first_column_name = "Coordination"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ2/coordinationExtractionTable", latex_table)
        



# =======================
# RQ 3 
# =======================
        
    def autonomyTable(self):
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
        
        caption = "Levels of Autonomy in Studies"
        label = "rq3-autonomy"
        tabular_size = "p{3.5cm} l p{15cm}"
        first_column_name = "Autonomy LVL"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ3/autonomyTable", latex_table)

        
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
        
        caption = "Level of Integration of Constituents in Studies"
        label = "rq3-lvl-integration"
        tabular_size = "p{3.5cm} l p{15cm}"
        first_column_name = "Integration LVL"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ3/levelOfIntegrationTable", latex_table)

     
     
# =======================
# RQ 4 
# =======================   
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
        
        caption = "SoS Type in Studies"
        label = "sos-type"
        tabular_size = "p{2.5cm} l p{14cm}"
        first_column_name = "SoS"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ4/sosTypeTable", latex_table)
        
        
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
        
        caption = "Emergence Type in Studies"
        label = "emergence-type"
        tabular_size = "p{2.5cm} l p{14cm}"
        first_column_name = "Emergence"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ4/emergenceTable", latex_table)
        
        
            
# =======================
# RQ 5 
# =======================
    def standardsTable(self, threshold = 2):
        df = self.df.copy()
    
        standards_column = "Standards Used (Cleaned Up)"
        citation_column = "Citation Code"

        df["Paper ID"] = ["T{:02d}".format(i + 1) for i in range(len(df))]
        
        # Helper function to format citations: Remove nulls and duplicates, then wrap each citation individually with \citepPS{...}.
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
            
        
        caption = "Standards Used in Papers"
        label = "standards"
        tabular_size = "p{5cm} l p{11.5cm}"
        first_column_name = "Standard"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ5/standards", latex_table)
        
        
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
        
        caption = "TRL in Studies"
        label = "trl"
        tabular_size = "p{2.5cm} l p{14cm}"
        first_column_name = "TRL"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ5/trlTable", latex_table)
        
        
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
        
        caption = "Evaluation in Studies"
        label = "rq5-evaluation"
        tabular_size = "p{2.5cm} l p{14cm}"
        first_column_name = "Evaluation"
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex("RQ5/EvaluationTable", latex_table)

                
                           
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

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('future.no_silent_downcasting', True)
data_path = "data/Data extraction sheet.xlsx"
results_path = "./output/tables"

class Analysis:
    observation_map = {
        1: "motivationsTable", #RQ1
        2: "intentsTable", #RQ1
        3: "domainsTable", #RQ1
        4: "topologyExtractionTable", #RQ2
        5: "coordinationExtractionTable", #RQ2
        6: "constituentUnitsTable", #RQ2
        7: "autonomyTable", #RQ3
        8: "levelOfIntegrationTable", #RQ3
        9: "emergenceTable", #RQ4
        10: "sosTypeTable", #RQ4
        11: "trlTable", #RQ5
        12: "EvaluationTable", #RQ5
        13: "standardsTable", #RQ5
        14: "contributionTypeTable", #RQ5
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
        df["Paper ID"] = [f"T{i+1:02d}" for i in range(len(df))]
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
        
        
    def generate_summary_table(self, column, caption, label, tabular_size, first_column_name, save_location):
        df = self.df.copy()
        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=("Citation Code", lambda x: ", ".join(f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()) if x.dropna().any() else "\\citepPS{placeholder}")
        ).reset_index().sort_values(by="Paper_Count", ascending=False)

        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex(f"{save_location}", latex_table)

      
# =======================
# RQ 1 
# =======================          
    def motivationsTable(self):
        self.generate_summary_table("Motivation (Clustered)", "Motivations in Studies", "motivations", "p{5cm} l p{12.5cm}", "Motivation", "RQ1/motivations")
        
    def intentsTable(self):
        self.generate_summary_table("Intent", "Intents in Studies", "rq1-intent", "p{5cm} l p{12.5cm}", "Intent", "RQ1/intentsTable")
        
    def domainsTable(self):
        self.generate_summary_table("Domain (Aggregated)", "Domains of Studies", "rq1-domains", "p{5cm} l p{12.5cm}", "Domain", "RQ1/domainsTable")

# =======================
# RQ 2
# =======================
    def topologyExtractionTable(self):
        df = self.df.copy()
        topology_categories = ["Hierarchical", "Centralized", "Decentralized", "Distributed", "Federated"]
        df["Extracted Topologies"] = df["Topology of DT/PT"].apply(lambda x: list(set(topology_categories).intersection(re.findall(r"\b\w+\b", str(x)))) if pd.notna(x) else [])
        summary_df = df.explode("Extracted Topologies").groupby("Extracted Topologies").agg(
            Paper_Count=("Paper ID", "count"),
            Citations=("Citation Code", lambda x: ", ".join(f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()) if x.dropna().any() else "\\citepPS{placeholder}")
        ).reset_index().sort_values(by="Paper_Count", ascending=False)
        latex_table = self.generate_latex_table(summary_df, "Topologies in Studies", "rq2-topology", "p{3.5cm} l p{15cm}", "Topology")
        self.saveLatex("RQ2/topologyExtractionTable", latex_table)

    def coordinationExtractionTable(self):
        self.generate_summary_table("Coordination (Cleaned)", "Coordination in Studies", "rq2-coordination", "p{3.5cm} l p{15cm}", "Coordination", "RQ2/coordinationExtractionTable")
    
    def constituentUnitsTable(self):
        self.generate_summary_table("Constituent unit (Aggregated)", "Constituent Units in Studies", "rq2-constituent-units", "p{5cm} l p{12.5cm}", "Constituent Unit", "RQ2/constituentUnitsTable")
    
# =======================
# RQ 3 
# =======================
    def autonomyTable(self):
        self.generate_summary_table("DT Class", "Levels of Autonomy in Studies", "rq3-autonomy", "p{3.5cm} l p{15cm}", "Autonomy LVL", "RQ3/autonomyTable")

    def levelOfIntegrationTable(self):
        self.generate_summary_table("Level of Integration (Cleaned)", "Level of Integration in Studies", "rq3-lvl-integration", "p{3.5cm} l p{15cm}", "Integration LVL", "RQ3/levelOfIntegrationTable")


    
# =======================
# RQ 4 
# =======================   
    def sosTypeTable(self):
            self.generate_summary_table("Type of SoS", "SoS Type in Studies", "sos-type", "p{2.5cm} l p{14cm}", "SoS", "RQ4/sosTypeTable")

    def emergenceTable(self):
        self.generate_summary_table("Emergence", "Emergence Type in Studies", "emergence-type", "p{2.5cm} l p{14cm}", "Emergence", "RQ4/emergenceTable")
            
# =======================
# RQ 5 
# =======================
    def trlTable(self):
        self.generate_summary_table("TRL", "TRL in Studies", "trl", "p{2.5cm} l p{14cm}", "TRL", "RQ5/trlTable")

    def EvaluationTable(self):
        self.generate_summary_table("Evaluation", "Evaluation in Studies", "rq5-evaluation", "p{2.5cm} l p{14cm}", "Evaluation", "RQ5/EvaluationTable")
        
    def contributionTypeTable(self):
        self.generate_summary_table("Contribution type", "Contribution Type in Studies", "rq5-contribution-type", "p{5cm} l p{12.5cm}", "Contribution Type", "RQ5/contributionTypeTable")
                  
    def standardsTable(self, threshold = 2):
        df = self.df.copy()
    
        standards_column = "Standards Used (Cleaned Up)"
        citation_column = "Citation Code"

        
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
        
        
                     
# =======================
# Saving and Running Script 
# =======================        
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

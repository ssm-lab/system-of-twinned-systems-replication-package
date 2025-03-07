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
        17: "dtServicesTable", #RQ3
        20: "programmingLangaugesTables", #RQ2
        19: "frameworksTables", #RQ3
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
        
    # For simple tables with one item per row  
    def generate_summary_table(self, column, caption, label, tabular_size, first_column_name, save_location):
        df = self.df.copy()
        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=("Citation Code", lambda x: ", ".join(f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()) if x.dropna().any() else "\\citepPS{placeholder}")
        ).reset_index().sort_values(by="Paper_Count", ascending=False)

        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex(f"{save_location}", latex_table)
        
    # For tables with multiple items per row seperated by a delimiter
    def generate_delimiter_table(self, column, caption, label, tabular_size, first_column_name, save_location, delimiter=","):
        df = self.df.copy()
        citation_column = "Citation Code"
        rows = []

        for _, row in df.iterrows():
            main_values = row[column]
            if pd.isna(main_values):
                continue
            # Split values and strip whitespace
            main_values_list = [s.strip() for s in str(main_values).split(delimiter) if s.strip()]
            for val in main_values_list:
                rows.append({
                    "Value": val,
                    "Paper ID": row["Paper ID"],
                    "Citation Code": row[citation_column] if citation_column in row else None
                })

        # Create an exploded DataFrame
        exploded_df = pd.DataFrame(rows)

        # Group by unique values and aggregate the number of studies and citations
        summary_df = exploded_df.groupby("Value").agg(
            Paper_Count=("Paper ID", "nunique"),
            Citations=(citation_column, lambda x: ", ".join([f"\\citepPS{{{cite}}}" 
                                                            for cite in x.dropna().unique()]) 
                                            if not x.dropna().empty else "\\citepPS{placeholder}")
        ).reset_index()

        # Sort by Paper_Count in descending order
        summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

        # Generate LaTeX table
        latex_table = self.generate_latex_table(summary_df, caption, label, tabular_size, first_column_name)
        self.saveLatex(save_location, latex_table)
        
    # for tables with multiple items per row and need an other category based on a frequency threshold
    def generate_other_cat_table(self, group_by_col, latex_caption, latex_label, latex_tabular_size, latex_first_column, latex_filename, delimiter = ", ", threshold=2):
        df = self.df.copy()
        citation_col = "Citation Code"
        count_col = "Paper ID"

        # Helper function to format citations
        def format_citations(citations):
            unique = {cite for cite in citations if pd.notna(cite)}
            return ", ".join(f"\\citepPS{{{c}}}" for c in unique) if unique else "\\citepPS{placeholder}"

        # Extract and explode values by delimiter
        rows = [
            {group_by_col: value.strip(), "Paper ID": row[count_col], "Citation Code": row[citation_col]}
            for _, row in df.iterrows() if pd.notna(row[group_by_col])
            for value in str(row[group_by_col]).split(delimiter) if value.strip()
        ]

        exploded_df = pd.DataFrame(rows)

        # Aggregate data
        agg_funcs = {"Paper_Count": ("Paper ID", "nunique")}
        if citation_col:
            agg_funcs["Citations"] = ("Citation Code", lambda x: format_citations(x.dropna()))

        summary_df = exploded_df.groupby(group_by_col).agg(**agg_funcs).reset_index()

        # Identify low-frequency items
        mask = summary_df["Paper_Count"] <= threshold

        if mask.all():  # If ALL items fall into "Other", keep them listed individually
            summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)
        elif mask.any():  # If some, but not all, items are below the threshold, group them into "Other"
            other_citations = exploded_df.loc[exploded_df[group_by_col].isin(summary_df[mask][group_by_col]), citation_col] if citation_col else None
            summary_df = summary_df[~mask].sort_values(by="Paper_Count", ascending=False)

            # Create "Other" category
            other_row = {
                group_by_col: "Other",
                "Paper_Count": other_citations.nunique() if citation_col else mask.sum(),
                "Citations": format_citations(other_citations.dropna()) if citation_col else None
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([other_row])], ignore_index=True)

        # Generate and save LaTeX table
        latex_table = self.generate_latex_table(summary_df, latex_caption, latex_label, latex_tabular_size, latex_first_column)
        self.saveLatex(latex_filename, latex_table)

        
        
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
        self.generate_summary_table("Constituent unit (higher level aggregation)", "Constituent Units in Studies", "rq2-constituent-units", "p{5cm} l p{12.5cm}", "Constituent Unit", "RQ2/constituentUnitsTable")
        
    def programmingLangaugesTables(self):        
        self.generate_delimiter_table(
        column="Programming Languages (General Purpose)", 
        caption="Programming Languages Used in Papers", 
        label="rq2-programming-language", 
        tabular_size="p{5cm} l p{11.5cm}", 
        first_column_name="Programming Language", 
        save_location="RQ2/generalProgrammingLanguagesTable"
        )
        
        self.generate_delimiter_table(
            column="Programming Languages (Markup, Styling)", 
            caption="Styling and Markup Programming Languages Used in Papers", 
            label="rq2-markup-styling-programming-language", 
            tabular_size="p{5cm} l p{11.5cm}", 
            first_column_name="Programming Language", 
            save_location="RQ2/markupStylingProgrammingLanguagesTable"
        )
        
        self.generate_delimiter_table(
        column="Programming Languages (Data Representation)", 
        caption="Data Representation Formats Used in Papers", 
        label="rq2-data-representation-formats", 
        tabular_size="p{5cm} l p{11.5cm}", 
        first_column_name="Data Format", 
        save_location="RQ2/dataRepresentationFormatTable"
    )

        
    def frameworksTables(self):
        threshold = 1
        delimiter = ", "

        # Dictionary mapping column names to LaTeX labels and filenames
        categories = {
            "Digital Twin & IoT": ("DT and IoT Frameworks used in Studies", "rq2-frameworks-dt-iot", "RQ2/frameworks_tables/dtIot"),
            "Modeling & Simulation": ("Modeling and Simulation Frameworks", "rq2-frameworks-modeling", "RQ2/frameworks_tables/modeling"),
            "AI, Data Analytics & Machine Learning": ("AI and Data Analytics Frameworks", "rq2-frameworks-ai", "RQ2/frameworks_tables/ai"),
            "Cloud, Edge, and DevOps": ("Cloud and Edge Frameworks", "rq2-frameworks-cloud", "RQ2/frameworks_tables/cloud"),
            "Systems Engineering & Architecture": ("Systems Engineering Frameworks", "rq2-frameworks-systems", "RQ2/frameworks_tables/systems"),
            "Data Management": ("Data Management Frameworks", "rq2-frameworks-data", "RQ2/frameworks_tables/data"),
            "Geospatial & Visualization Technologies": ("Geospatial and Visualization Frameworks", "rq2-frameworks-geo", "RQ2/frameworks_tables/geo"),
            "Application Development & Web Technologies": ("App Development and Web Frameworks", "rq2-frameworks-appdev", "RQ2/frameworks_tables/appdev"),
        }

        # Loop through each category and generate the LaTeX table
        for column, (caption, label, filename) in categories.items():
            self.generate_other_cat_table(
                group_by_col=column,
                latex_caption=caption,
                latex_label=label,
                latex_tabular_size="p{3.5cm} l p{15cm}",
                latex_first_column="Tool",
                latex_filename=filename,
                delimiter=delimiter,
                threshold=threshold,
            )

    
# =======================
# RQ 3 
# =======================
    def autonomyTable(self):
        self.generate_summary_table("DT Class", "Levels of Autonomy in Studies", "rq3-autonomy", "p{3.5cm} l p{15cm}", "Autonomy LVL", "RQ3/autonomyTable")

    def levelOfIntegrationTable(self):
        self.generate_summary_table("Level of Integration (Cleaned)", "Level of Integration in Studies", "rq3-lvl-integration", "p{3.5cm} l p{15cm}", "Integration LVL", "RQ3/levelOfIntegrationTable")
        
    def dtServicesTable(self):
        self.generate_delimiter_table(
        column="Services (Cleaned)", 
        caption="DT Services Used in Papers", 
        label="rq3-dt-services", 
        tabular_size="p{5cm} l p{11.5cm}", 
        first_column_name="Service", 
        save_location="RQ3/dtServicesTable"
        )


    
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
                  
    def standardsTable(self, threshold=2):
        df = self.df.copy()
        standards_column = "Standards Used (Cleaned Up)"
        citation_column = "Citation Code"

        def format_citations(citations):
            unique = {cite for cite in citations if pd.notna(cite)}
            return ", ".join(f"\\citepPS{{{c}}}" for c in unique) if unique else "\\citepPS{placeholder}"

        rows = [
            {"Standard": std.strip(), "Paper ID": row["Paper ID"], "Citation Code": row[citation_column]}
            for _, row in df.iterrows() if pd.notna(row[standards_column])
            for std in str(row[standards_column]).split(";") if std.strip()
        ]

        exploded_df = pd.DataFrame(rows)

        # Aggregate by frequency
        summary_df = exploded_df.groupby("Standard").agg(
            Paper_Count=("Paper ID", "nunique"),
            Citations=(citation_column, lambda x: format_citations(x.dropna()))
        ).reset_index()

        # Create other category for low frequency items
        mask = summary_df["Paper_Count"] <= threshold
        if mask.any():
            other_citations = exploded_df.loc[exploded_df["Standard"].isin(summary_df[mask]["Standard"]), citation_column]
            summary_df = summary_df[~mask].sort_values(by="Paper_Count", ascending=False)
            summary_df = pd.concat([
                summary_df,
                pd.DataFrame([{
                    "Standard": "Other",
                    "Paper_Count": other_citations.nunique(),
                    "Citations": format_citations(other_citations.dropna())
                }])
            ], ignore_index=True)
        
        latex_table = self.generate_latex_table(summary_df, "Standards Used in Papers", "standards", "p{5cm} l p{11.5cm}", "Standard")
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

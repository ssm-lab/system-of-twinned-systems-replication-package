import argparse
from collections import defaultdict
import os
import pandas as pd
import matplotlib.pyplot as plt

__author__ = "Feyi Adesanya"
__copyright__ = "Copyright 2024, Sustainable Systems and Methods Lab (SSM)"
__license__ = "GPL-3.0"

pd.set_option('future.no_silent_downcasting', True)
data_path = "data/Data extraction sheet.xlsx"
results_path = "./output/tables"

class Analysis:
    observation_map = {
        1: "motivationsTable", #RQ1
        2: "intentsTable", #RQ1
        3: "domainsTable", #RQ1
        4: "topologyExtractionTable", #RQ2
        22: "spatialDistributionTable", #RQ2
        5: "coordinationExtractionTable", #RQ2
        6: "constituentUnitsTable", #RQ2
        7: "autonomyTable", #RQ3
        9: "emergenceTable", #RQ4
        10: "sosTypeTable", #RQ4
        11: "trlTable", #RQ5
        13: "standardsTable", #RQ5
        14: "contributionTypeTable", #RQ5
        15: "dtServicesTable", #RQ3
        16: "programmingLangaugesTables", #RQ2
        17: "generate_frameworks_table", #RQ3
        18: "dtOrSoSRelated", # RQ5
        20: "securityTable", 
        21: "reliabilityTable",
        23: "generate_structured_eval_table",
        24: "generate_formalisms_methods_table",
        25: "generate_challenges_table",
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
            \\multicolumn{{1}}{{c}}{{\\textbf{{Frequency}}}} & 
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
    def generate_summary_table(self, column, caption, label, tabular_size, first_column_name, save_location, custom_order=None):
        df = self.df.copy()
        summary_df = df.groupby(column).agg(
            Paper_Count=("Paper ID", "count"),
            Citations=("Citation Code", lambda x: ", ".join(f"\\citepPS{{{cite}}}" for cite in x.dropna().unique()) if x.dropna().any() else "\\citepPS{placeholder}")
        ).reset_index()
        
        if custom_order:
            summary_df[column] = pd.Categorical(summary_df[column], categories=custom_order, ordered=True)
            summary_df = summary_df.sort_values(by=column)

        else:
            summary_df = summary_df.sort_values(by="Paper_Count", ascending=False)

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

        # Extract and explode values by delimiter if it exist
        rows = []
        for _, row in df.iterrows():
            cell_value = row[group_by_col]
            if pd.isna(cell_value):
                continue

            if delimiter is None:
                values = [str(cell_value).strip()]
            else:
                values = [v.strip() for v in str(cell_value).split(delimiter) if v.strip()]

            for value in values:
                rows.append({
                    group_by_col: value,
                    "Paper ID": row[count_col],
                    "Citation Code": row.get(citation_col)
                })


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
        
        
    def generate_hierarchical_table(self, category_list, caption, label, filename, column_label, threshold=2, latex_friendly_names=None):
        df = self.df.copy()
        citation_col = "Citation Code"

        hierarchy = defaultdict(lambda: defaultdict(lambda: {"citations": set(), "count": 0}))

        for _, row in df.iterrows():
            citation = row[citation_col]
            for category in category_list:
                if pd.isna(row.get(category)):
                    continue
                submethods = [s.strip() for s in str(row[category]).split(",") if s.strip()]
                for method in submethods:
                    hierarchy[category][method]["count"] += 1
                    if pd.notna(citation):
                        hierarchy[category][method]["citations"].add(citation)

        # Start building LaTeX
        latex_lines = [
            "\\begin{table*}[]",
            "\\centering",
            "\\setlength{\\tabcolsep}{1em}",
            f"\\caption{{{caption}}}",
            f"\\label{{tab:{label}}}",
            "\\footnotesize",
            "\\begin{tabular}{@{}p{5.0cm} l p{9cm}@{}}", 
            "\\toprule",
            f"\\textbf{{{column_label}}} & \\textbf{{Frequency}} & \\textbf{{Studies}} \\\\",
            "\\midrule"
        ]


        category_totals = []
        for category in category_list:
            submethods = hierarchy.get(category, {})
            all_citations = set().union(*(v["citations"] for v in submethods.values()))
            category_totals.append((category, len(all_citations)))
            
        sorted_categories = sorted(category_totals, key=lambda x: x[1], reverse=True)
        for category, total_cites in sorted_categories:
            submethods = hierarchy.get(category, {})
            if not submethods:
                continue

            above = {k: v for k, v in submethods.items() if v["count"] >= threshold}
            below = {k: v for k, v in submethods.items() if v["count"] < threshold}

            label_name = latex_friendly_names.get(category, category) if latex_friendly_names else category
            latex_lines.append(f"\\textbf{{{label_name}}} & \\textbf{{\\maindatabar{{{total_cites}}}}} & \\\\")

            for method, data in sorted(above.items(), key=lambda item: len(item[1]["citations"]), reverse=True):
                count = len(data["citations"])
                cites = ", ".join(f"\\citepPS{{{c}}}" for c in sorted(data["citations"]))
                latex_lines.append(f"\\;\\;\\corner{{}} {method} & \\subdatabar{{{count}}} & {cites} \\\\")

            if below:
                all_below_cites = set().union(*[v["citations"] for v in below.values()])
                count = len(all_below_cites)
                cites = ", ".join(f"\\citepPS{{{c}}}" for c in sorted(all_below_cites))
                latex_lines.append(f"\\;\\;\\corner{{}} \\textit{{Other}} & \\subdatabar{{{count}}} & {cites} \\\\")

        latex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
        self.saveLatex(filename, "\n".join(latex_lines))


        
        
# =======================
# RQ 1 
# =======================          
    def motivationsTable(self):
        self.generate_summary_table("Motivation (Clustered)", "Motivations for Combining DT and SoS", "motivations-table", "p{2.5cm} l p{15cm}", "Motivation", "rq1/motivations")
        
    def intentsTable(self):
        self.generate_summary_table("Intent", "Intents of Combining DT and SoS", "intents-table", "p{4cm} l p{13.5cm}", "Intent", "rq1/intentsTable")
        
    def domainsTable(self):
        self.generate_other_cat_table(
            group_by_col="Domain (Aggregated)",
            latex_caption="Application Domains",
            latex_label="domains-table",
            latex_tabular_size="p{4cm} l p{13.5cm}",
            latex_first_column="Domain",
            latex_filename="rq1/domainsTable",
            delimiter=None,
            threshold=2,
        )
        
    def generate_challenges_table(self):
        df = pd.read_excel(data_path, sheet_name="Challenges Separated")
        eval_col = "Challenges"
        expanded_col = "Further Categorization"
        citation_col = "Citation Code"

        hierarchy = {}

        for _, row in df.iterrows():
            eval_type = str(row[eval_col]).title() if pd.notna(row[eval_col]) else None
            if pd.isna(eval_type):
                continue

            expanded_items = [s.strip().title() for s in str(row[expanded_col]).split(",") if s.strip()]
            citation = row[citation_col]

            if eval_type not in hierarchy:
                hierarchy[eval_type] = {}

            for item in expanded_items:
                if item not in hierarchy[eval_type]:
                    hierarchy[eval_type][item] = set()
                if pd.notna(citation):
                    hierarchy[eval_type][item].add(citation)

        # Start LaTeX table
        latex_lines = [
            "\\begin{table*}[]",
            "\\centering",
            "\\setlength{\\tabcolsep}{1em}",
            "\\caption{Challenges}",
            "\\label{tab:challenges-table}",
            "\\footnotesize",
            "\\begin{tabular}{@{}p{5.25cm} l p{8cm}@{}}", 
            "\\toprule",
            "\\textbf{Challenge} & \\textbf{Frequency} & \\textbf{Studies} \\\\",
            "\\midrule"
        ]
        
        sorted_hierarchy = sorted(
            hierarchy.items(),
            key=lambda item: len(set().union(*item[1].values())),
            reverse=True
        )

        for eval_type, submethods in sorted_hierarchy:
            total_cites = set().union(*submethods.values())
            total_count = len(total_cites)
            # Top-level row: no citations
            latex_lines.append(f"\\textbf{{{eval_type}}} & \\textbf{{\maindatabar{{{total_count}}}}} & \\\\")

            # Submethods with citations
            for method, citations in sorted(submethods.items(), key=lambda item: len(item[1]), reverse=True):
                count = len(citations)
                citation_str = ", ".join(f"\\citepPS{{{c}}}" for c in sorted(citations))
                latex_lines.append(f"\\;\;\\corner{{}} {method} & \subdatabar{{{count}}} & {citation_str} \\\\")

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}"
        ])

        self.saveLatex("rq1/challenges_table", "\n".join(latex_lines))
    
        

# =======================
# RQ 2
# =======================
    def topologyExtractionTable(self):        
        self.generate_summary_table("Topology of DT/PT (Cleaned)", "Topologies", "topology-table", "p{2.5cm} l p{15cm}", "Topology", "rq2/topologyExtractionTable")

    def spatialDistributionTable(self):
        self.generate_summary_table("Spatial Distribution", "Spatial Distribution", "spatial-distribution-table", "p{3.5cm} l p{15cm}", "Distribution", "rq2/spatialDistributionTable")
        
    def coordinationExtractionTable(self):
        self.generate_summary_table("Coordination (Cleaned)", "Coordination", "coordination-table", "p{3.5cm} l p{15cm}", "Coordination", "rq2/coordinationExtractionTable")
    
    def constituentUnitsTable(self):
        self.generate_summary_table("Constituent unit (higher level aggregation)", "Constituent Units", "constituent-units-table", "p{5cm} l p{12.5cm}", "Constituent Unit", "rq2/constituentUnitsTable")

    
# =======================
# RQ 3 
# =======================
    def autonomyTable(self):
        self.generate_summary_table("DT Class", "Levels of Autonomy", "autonomy-table", "p{5cm} l p{13.5cm}", "Autonomy", "rq3/autonomyTable")

    def dtServicesTable(self):
        self.generate_delimiter_table(
        column="Services (Cleaned)", 
        caption="DT Services Used in Papers", 
        label="dt-services-table", 
        tabular_size="p{3.5cm} l p{14cm}", 
        first_column_name="Service", 
        save_location="rq3/dtServicesTable"
        )


    def generate_formalisms_methods_table(self, threshold=2):
        self.generate_hierarchical_table(
            category_list=[
                "Mathematical and Statistical",
                "Formal and State Based Methods",
                "Discrete-Event Simulation",
                "Continuous Simulation",
                "Agent-Based Simulation",
                "Ontological and Knowledge Representation",
                "Architectural and Structural",
                "Spatial and Visual Modelling",
                "AI and Machine Learning"
            ],
            caption="Modeling and Simulation Formalisms",
            label="modeling-methods-structured-table",
            filename="rq3/hierarchicalModelingMethodsTable",
            column_label="Category",
            threshold=threshold
        )


    
# =======================
# RQ 4 
# =======================   
    def sosTypeTable(self):
            self.generate_summary_table("Type of SoS", "SoS Type", "sos-type-table", "p{2.5cm} l p{14cm}", "SoS", "rq4/sosTypeTable")

    def emergenceTable(self):
        self.generate_summary_table("Emergence", "Emergence Type", "emergence-type-table", "p{1.5cm} l p{15cm}", "Emergence", "rq4/emergenceTable")
            
    
# =======================
# RQ 5
# =======================
    def securityTable(self):
        custom_order = [
            "Not Mentioned",
            "Mentioned",
            "Architecturally Addressed",
            "Explicitly Modelled",
            "Evaluated or Validated"
        ]
        self.generate_summary_table("Security/Confidentiality Level", "Security", "security-table", "p{4cm} l p{13.5cm}", "Context", "rq5/securityTable", custom_order)
        
    def reliabilityTable(self):
        custom_order = [
            "Not Mentioned",
            "Mentioned",
            "Architecturally Addressed",
            "Explicitly Modelled",
            "Evaluated or Validated"
        ]
        self.generate_summary_table("Reliability Level", "Reliability", "reliability-table", "p{4cm} l p{13.5cm}", "Context", "rq5/reliabilityTable", custom_order)
     
     
# =======================
# RQ 6
# =======================
    def trlTable(self):
        custom_order = [
            "Initial",
            "Proof-of-Concept",
            "Demo Prototype",
            "Deployed Prototype",
            "Operational"
        ]
        self.generate_summary_table("TRL", "TRL", "trl-table", "p{3.5cm} l p{15cm}", "TRL", "rq6/trlTable", custom_order)            
            
    def generate_structured_eval_table(self):
        df = self.df.copy()
        eval_col = "Evaluation"
        expanded_col = "Eval/Val Expanded"
        citation_col = "Citation Code"

        hierarchy = {}

        for _, row in df.iterrows():
            eval_type = str(row[eval_col]).title() if pd.notna(row[eval_col]) else None
            if pd.isna(eval_type):
                continue

            expanded_items = [s.strip().title() for s in str(row[expanded_col]).split(",") if s.strip()]
            citation = row[citation_col]

            if eval_type not in hierarchy:
                hierarchy[eval_type] = {}

            for item in expanded_items:
                if item not in hierarchy[eval_type]:
                    hierarchy[eval_type][item] = set()
                if pd.notna(citation):
                    hierarchy[eval_type][item].add(citation)

        # Start LaTeX table
        latex_lines = [
            "\\begin{table*}[]",
            "\\centering",
            "\\setlength{\\tabcolsep}{1em}",
            "\\caption{Validation and Evaluation Approaches}",
            "\\label{tab:evaluation-structured-table}",
            "\\footnotesize",
            "\\begin{tabular}{@{}p{4.0cm} l p{10cm}@{}}", 
            "\\toprule",
            "\\textbf{Evaluation Category} & \\textbf{Frequency} & \\textbf{Studies} \\\\",
            "\\midrule"
        ]

        for eval_type, submethods in hierarchy.items():
            total_cites = set().union(*submethods.values())
            total_count = len(total_cites)
            # Top-level row: no citations
            latex_lines.append(f"\\textbf{{{eval_type}}} & \\textbf{{\maindatabar{{{total_count}}}}} & \\\\")

            # Submethods with citations
            for method, citations in sorted(submethods.items(), key=lambda item: len(item[1]), reverse=True):
                count = len(citations)
                citation_str = ", ".join(f"\\citepPS{{{c}}}" for c in sorted(citations))
                latex_lines.append(f"\\;\;\\corner{{}} {method} & \subdatabar{{{count}}} & {citation_str} \\\\")

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}"
        ])

        self.saveLatex("rq6/hierarchicalEvaluationTable", "\n".join(latex_lines))

        
    def contributionTypeTable(self):
        self.generate_summary_table("Contribution type", "Contribution Type", "contribution-type-table", "p{2cm} l p{15.5cm}", "Contribution", "rq6/contributionTypeTable")
                  
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
        
        latex_table = self.generate_latex_table(summary_df, "Standards", "standards-table", "p{6.5cm} l p{10cm}", "Standard")
        self.saveLatex("rq6/standards", latex_table)
        
    def dtOrSoSRelated(self):
        self.generate_summary_table("Do The Studies Use Standards in More of an SoS or DT context", "Standards Usage Context (DT vs. SoS)", "dt-or-sos-related-table", "p{2cm} l p{15.5cm}", "Context", "rq6/dtOrSoSRelated")
        
        
# =======================
# RQ 7
# =======================   
    def programmingLangaugesTables(self, threshold=0):
        self.generate_hierarchical_table(
            category_list=[
                "General Purpose",
                "Markup and Styling",
                "Data Representation"
            ],
            caption="Programming Languages and Data Formats",
            label="programming-languages-structured-table",
            filename="rq7/hierarchicalProgrammingLanguagesTable",
            column_label="Category",
            threshold=threshold
        )

    
    def generate_frameworks_table(self, threshold=2):
        self.generate_hierarchical_table(
            category_list=[
                "Digital Twin & IoT", "Modeling & Simulation", "AI, Data Analytics & Machine Learning",
                "Cloud, Edge, and DevOps", "Systems Engineering & Architecture",
                "Data Management", "Geospatial & Visualization Technologies",
                "Application Development & Web Technologies"
            ],
            caption="Tools and Frameworks",
            label="frameworks-structured-table",
            filename="rq7/hierarchicalFrameworksTable",
            column_label="Category",
            threshold=threshold,
            latex_friendly_names={
                "Digital Twin & IoT": "Digital Twin \\& IoT",
                "Modeling & Simulation": "Modeling \\& Simulation",
                "AI, Data Analytics & Machine Learning": "AI, Data Analytics \\& ML",
                "Cloud, Edge, and DevOps": "Cloud, Edge, and DevOps",
                "Systems Engineering & Architecture": "Systems Eng. \\& Architecture",
                "Data Management": "Data Management",
                "Geospatial & Visualization Technologies": "Geospatial \\& Visualization",
                "Application Development & Web Technologies": "App/Web Technologies",
            }
        )

                     
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

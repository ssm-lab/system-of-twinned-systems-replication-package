import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import plotly.graph_objects as go
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrow

__author__ = "Feyi Adesanya"
__copyright__ = "Copyright 2024, Sustainable Systems and Methods Lab (SSM)"
__license__ = "GPL-3.0"


pd.set_option('future.no_silent_downcasting', True)
data_path = "data/Data extraction sheet.xlsx"
results_path = "./output/figures"

class Analysis:
    observation_map = {
        1: "intentOfSoSDT", # RQ1
        2: "sosDimensions", # RQ4
        3: "trlVsContributionType", #RQ6
        4: "dtSosQuadrants"
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
# Overall
# =======================     
    def dtSosQuadrants(self):
        # Define quadrants
        quadrants = {
            "Digital Twins": (0, 1, 1, 1),
            "?": (1, 1, 1, 1),
            "Information \nSystems": (0, 0, 1, 1),
            "System \nof Systems": (1, 0, 1, 1)
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        # Draw each box
        for label, (x, y, w, h) in quadrants.items():
            color = "#1E3A5F" if label == "?" else "#B0B0B0"
            ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='white'))
            ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                    fontsize=14, fontweight='medium', color="#ffffff")

        # Draw x and y axis arrows
        arrowprops = dict(width=0.02, head_width=0.1, head_length=0.1, color='black')

        # x-axis
        ax.add_patch(FancyArrow(0, -0.1, 2.1, 0, **arrowprops))
        # y-axis
        ax.add_patch(FancyArrow(-0.1, 0, 0, 2.1, **arrowprops))

        # Axis labels
        ax.text(1, -0.25, "Loose Systems Coordination", ha='center', va='top', fontsize=12)
        ax.text(-0.3, 1, "Digital–Physical Convergence", ha='center', va='center',
                fontsize=12, rotation=90)

        # Formatting
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # plt.tight_layout()
        self.savefig("dtSosQuadrants")

      
# =======================
# RQ 1 
# =======================  
        
    def intentOfSoSDT(self):
        df = self.df.copy()
        intent_domain_counts = df.groupby(["Intent", "Domain (Aggregated)"]).size().reset_index(name="Count")
        pivot_df = intent_domain_counts.pivot(index="Domain (Aggregated)", columns="Intent", values="Count").fillna(0)
        dt_col, sos_col = pivot_df.columns[:2].tolist()


        # Group 1: SoS-exclusive (DT==0 and SoS>0)
        # Group 2: Common (DT>0 and SoS>0)
        # Group 3: DT-exclusive (DT>0 and SoS==0)
        pivot_df["order_group"] = 2
        pivot_df.loc[(pivot_df[dt_col] == 0) & (pivot_df[sos_col] > 0), "order_group"] = 1
        pivot_df.loc[(pivot_df[dt_col] > 0) & (pivot_df[sos_col] == 0), "order_group"] = 3


        pivot_df["difference"] = pivot_df[sos_col] - pivot_df[dt_col]
        pivot_df = pivot_df.sort_values(by=["order_group", "difference"], ascending=[True, False])
        y_categories = pivot_df.index.tolist()

        # Mirror the DT column for plotting (so that DT bars extend to the left)
        pivot_df["DT_mirrored"] = -pivot_df[dt_col]

        fig = go.Figure()
        # Trace for DT (mirrored to the left)
        fig.add_trace(go.Bar(
            y=y_categories,
            x=[pivot_df.loc[domain, "DT_mirrored"] for domain in y_categories],
            name=dt_col,
            orientation='h',
            # marker_color='rgb(222,45,38)',
            marker_color="#f05a50",
            text=[f"{pivot_df.loc[domain, dt_col]:.0f}" for domain in y_categories],
            textfont=dict(size=21, color="black", weight="bold"),
            textposition='inside',
            hovertemplate='Domain: %{y}<br>' + dt_col + ': %{text}<extra></extra>',
        ))

        # Trace for SoS (to the right)
        fig.add_trace(go.Bar(
            y=y_categories,
            x=[pivot_df.loc[domain, sos_col] for domain in y_categories],
            name=sos_col,
            orientation='h',
            # marker_color='rgb(49,130,189)',
            marker_color="#85d4ff",
            text=[f"{pivot_df.loc[domain, sos_col]:.0f}" for domain in y_categories],
            textfont=dict(size=21, color="black", weight="bold"),
            textposition='inside',
            hovertemplate='Domain: %{y}<br>' + sos_col + ': %{text}<extra></extra>',
        ))

        # Determine a common maximum for symmetry
        max_count = max(pivot_df[dt_col].max(), pivot_df[sos_col].max())

        fig.update_layout(
            title=dict(
                text="Intent by Domain",
                x=0.5,  # centers the title horizontally
                font=dict(color='black', size=28)
            ),
            barmode='overlay',
            bargap=0.1,
            xaxis=dict(
                tickvals=[-max_count, -max_count/2, 0, max_count/2, max_count],
                ticktext=[str(int(max_count)), str(int(max_count/2)), "0", str(int(max_count/2)), str(int(max_count))],
                title=dict(text="# of Studies", font=dict(color='black', size=24)),
                showgrid=True,
                tickfont=dict(color='black', size=21),
            ),
            yaxis=dict(
                title=dict(text="Domain", font=dict(color='black', size=24)),
                automargin=True,
                showgrid=True,
                tickfont=dict(color='black', size=21),
            ),
            width=1400,
            height=900,
            font=dict(size=18),
            plot_bgcolor="white",
            legend=dict(
                x=0.65, 
                y=1,
                font=dict(size=21, color='black'), 
                bgcolor="white", 
            )
        )

        file_path = os.path.join(results_path, "RQ1/intentOfSoSDT.png")
        fig.write_image(file_path, scale=2)
        
# =======================
# RQ 2
# =======================
        

# =======================
# RQ 3 
# =======================
    
        
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
        
        no_vals = percentages["No"]
        partial_vals = percentages["Partial"]
        yes_vals = percentages["Yes"]

        left_no = -no_vals - (partial_vals / 2)
        left_partial = -partial_vals / 2
        left_yes = partial_vals / 2

        fig, ax = plt.subplots(figsize=(30, len(percentages) * 1.5))
        plt.subplots_adjust(left=0.25, right=0.95) 
        y_pos = np.arange(len(percentages))
        
        ax.barh(y_pos, no_vals, left=left_no, color="#f05a50", label="No")
        ax.barh(y_pos, partial_vals, left=left_partial, color="#d8d8d8", label="Partial")
        ax.barh(y_pos, yes_vals, left=left_yes, color="#85d4ff", label="Yes")

        # ax.set_facecolor("#f5f5f5")
        ax.set_facecolor("#ffffff")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, color="gray", alpha=0.6)

        ax.set_yticks(y_pos)
        # ax.set_yticklabels(percentages.index, fontsize=40)
        # ax.tick_params(axis='y', labelsize=40, pad=20) 
        font_prop = font_manager.FontProperties(size=32)
        ax.set_yticklabels(percentages.index, fontproperties=font_prop)
        ax.tick_params(axis='y', pad=30)
        ax.set_xlabel("Percentage", fontsize=28)
        ax.set_title("SoS Dimensions", fontsize=36)

        ax.tick_params(axis='x', labelsize=28)

        ax.axvline(0, color='black', linewidth=0.5, linestyle=':')

        # left_lim = left_no.min() - 1.5
        # right_lim = (left_yes + yes_vals).max() + 1.5
        # ax.set_xlim(left_lim, right_lim)
        ax.set_xlim(-100, 100)

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.0f}%"))

        min_width = 5
        for i in range(len(percentages.index)):
            if no_vals.iloc[i] > 0:
                xpos = left_no.iloc[i] + no_vals.iloc[i] / 2
                ha = 'center' if no_vals.iloc[i] >= min_width else 'right'
                ax.text(xpos if no_vals.iloc[i] >= min_width else left_no.iloc[i] - 1, i,
                        f"{no_vals.iloc[i]:.2f}%", va='center', ha=ha,
                        color='black', fontsize=24, weight=550)

            if partial_vals.iloc[i] > 0:
                ax.text(0, i - 0.3 if partial_vals.iloc[i] < min_width else i,
                        f"{partial_vals.iloc[i]:.2f}%", va='center',
                        ha='center', color='black', fontsize=24, weight=550)

            if yes_vals.iloc[i] > 0:
                xpos = left_yes.iloc[i] + yes_vals.iloc[i] / 2
                ha = 'center' if yes_vals.iloc[i] >= min_width else 'left'
                ax.text(xpos if yes_vals.iloc[i] >= min_width else xpos + 1, i,
                        f"{yes_vals.iloc[i]:.2f}%", va='center', ha=ha,
                        color='black', fontsize=24, weight=550)


        ax.legend(fontsize=18, loc="lower right")
        self.savefig("sosDimensions", upper_folder="RQ4")
            
# =======================
# RQ 5
# =======================   
       
       
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

        contribution_types = ["Conceptual", "Technical", "Case study"]
        colors = {
            "Conceptual": "#f05a50",
            "Technical": "#85d4ff",
            "Case study": "#d8d8d8"
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
                    percent = (height / total_studies) * 100
                    label = f"{int(height)} ({percent:.2f}%)"

                    offset = 5
                    ax.annotate(
                                label,
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, offset),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=10.5, fontweight='bold')

        ax.set_ylabel("Number of Papers", fontsize=14)
        ax.set_xticks(x)
        
        
        # Total studies per TRL level
        trl_totals = df_unique.groupby("TRL")["Citation Code"].nunique().reindex(trl_order).fillna(0).astype(int)
        trl_percents = (trl_totals / total_studies * 100)

        # Create new x-axis labels with totals and percentages
        x_labels = [
            f"{trl}\n{count} ({percent:.2f}%)"
            for trl, count, percent in zip(trl_order, trl_totals, trl_percents)
        ]
        ax.set_xticklabels(x_labels, rotation=15, fontsize=14)
        ax.set_title("Contribution Types by TRL Level", fontsize=16, pad=15)
        ax.legend(title="Contribution Type", fontsize=12, title_fontsize=13)
        ax.set_ylim(0, pivot_df.values.max() + 10) 
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        self.savefig("trlVsContributionType", upper_folder="rq6")

# =======================
# RQ 7
# =======================   

               
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

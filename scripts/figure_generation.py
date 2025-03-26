import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import plotly.graph_objects as go
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import matplotlib.patheffects as patheffects


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('future.no_silent_downcasting', True)
data_path = "data/Data extraction sheet.xlsx"
results_path = "./output/figures"

class Analysis:
    observation_map = {
        1: "intentOfSoSDT", # RQ1
        2: "sosDimensions", # RQ4
        3: "sosTypeVsEmergence", #RQ4
        4: "trlVsContributionType", #RQ5
        5: "topologyVsIntent", #RQ2
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
        
        total_studies = pivot_df[[dt_col, sos_col]].sum().sum()


        fig = go.Figure()
        # Trace for DT (mirrored to the left)
        fig.add_trace(go.Bar(
            y=y_categories,
            x=[pivot_df.loc[domain, "DT_mirrored"] for domain in y_categories],
            name=dt_col,
            orientation='h',
            marker_color='rgb(222,45,38)',
            text=[f"{pivot_df.loc[domain, dt_col]:.0f}" for domain in y_categories],
            textfont=dict(size=21),
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
            textfont=dict(size=21),
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
        # Rename specific labels
        rename_map = {
            "Dynamic Reconfiguration": "Reconfiguration",
            "Autonomy of Constituents": "Autonomy",
            "Emergence of Behaviour": "Emergence"  # Spelling from your earlier plot
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
        
        ax.barh(y_pos, no_vals, left=left_no, color="#d62728", label="No")
        ax.barh(y_pos, partial_vals, left=left_partial, color="#f0ad4e", label="Partial")
        ax.barh(y_pos, yes_vals, left=left_yes, color="#2ca02c", label="Yes")

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

        left_lim = left_no.min() - 1.5
        right_lim = (left_yes + yes_vals).max() + 1.5
        ax.set_xlim(left_lim, right_lim)

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.0f}%"))

        min_width = 5
        for i in range(len(percentages.index)):
            if no_vals.iloc[i] > 0:
                xpos = left_no.iloc[i] + no_vals.iloc[i] / 2
                ha = 'center' if no_vals.iloc[i] >= min_width else 'right'
                ax.text(xpos if no_vals.iloc[i] >= min_width else left_no.iloc[i] - 1, i,
                        f"{no_vals.iloc[i]:.2f}%", va='center', ha=ha,
                        color='white', fontsize=21, fontweight="bold", path_effects=[patheffects.withStroke(linewidth=1, foreground='black')])

            if partial_vals.iloc[i] > 0:
                ax.text(0, i - 0.3 if partial_vals.iloc[i] < min_width else i,
                        f"{partial_vals.iloc[i]:.2f}%", va='center',
                        ha='center', color='white', fontsize=21, fontweight="bold", path_effects=[patheffects.withStroke(linewidth=1, foreground='black')])

            if yes_vals.iloc[i] > 0:
                xpos = left_yes.iloc[i] + yes_vals.iloc[i] / 2
                ha = 'center' if yes_vals.iloc[i] >= min_width else 'left'
                ax.text(xpos if yes_vals.iloc[i] >= min_width else xpos + 1, i,
                        f"{yes_vals.iloc[i]:.2f}%", va='center', ha=ha,
                        color='white', fontsize=21, fontweight="bold", path_effects=[patheffects.withStroke(linewidth=1, foreground='black')])


        ax.legend(fontsize=18, loc="lower right")
        self.savefig("sosDimensions", upper_folder="RQ4")

        
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
                
    def trlVsContributionType(self):
        df = self.df.copy()

        trl_order = ["Initial", "Proof-of-Concept", "Demo prototype", "Deployed prototype", "Operational"]
        df["TRL"] = pd.Categorical(df["TRL"], categories=trl_order, ordered=True)

        trl_contribution_counts = df.groupby(["TRL", "Contribution type"]).size().reset_index(name="Count")

        pivot_df = trl_contribution_counts.pivot(index="TRL", columns="Contribution type", values="Count").fillna(0)

        pivot_df = pivot_df.reindex(trl_order[::-1])  # Reverse order to put Initial at the top

        trl_types = pivot_df.index.tolist()
        contribution_types = pivot_df.columns.tolist()

        colors = ["#3182bd", "#b5bbc3", "#de2d26"]

        num_contributions = len(contribution_types)
        bar_height = 0.7 / num_contributions 
        y_positions = np.arange(len(trl_types))
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

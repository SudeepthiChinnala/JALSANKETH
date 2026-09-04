"""
================================================================================
JAL SANKETH - EXPLORATORY DATA ANALYSIS (EDA)
Notebook Script: notebooks/01_data_exploration.py
================================================================================

This script performs comprehensive exploratory data analysis on the raw demo
weather dataset (data/raw/weather_demo.csv) without modifying the original data.

Checks:
  1. Dataset Shape & Dimensions
  2. Missing Values Analysis
  3. Duplicate Rows Check
  4. Data Types & Temporal Continuity
  5. Statistical Summary (Five-Number Summary, Skewness)
  6. Distribution of Rainfall (Zero-Inflation, Tail Analysis)
  7. Distribution of Heavy Rain Binary Target
  8. Multivariable Correlation Matrix & Meteorological Drivers
  9. Outlier Detection using Interquartile Range (IQR) & Boxplots

Saved Figures Location: reports/figures/
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 150


def run_eda(
    data_path: str = "data/raw/weather_demo.csv",
    figures_dir: str = "reports/figures"
) -> None:
    """
    Executes full EDA pipeline and generates publication-grade analysis plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 80)
    print("        JAL SANKETH - EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 80)
    print(f"[*] Ingesting dataset from : {data_path}")
    print(f"[*] Saving figures to       : {figures_dir}")
    print("-" * 80)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Target data file not found at '{data_path}'. Please run data/create_demo_dataset.py first.")

    # Read data without modifying original file
    df = pd.read_csv(data_path)

    # --------------------------------------------------------------------------
    # 1. SHAPE & DIMENSIONS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. DATASET SHAPE & DIMENSIONS")
    print("=" * 50)
    n_rows, n_cols = df.shape
    print(f"  * Total Records (Rows) : {n_rows:,}")
    print(f"  * Total Features (Cols): {n_cols}")
    print(f"  * Memory Usage         : {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

    # --------------------------------------------------------------------------
    # 2. MISSING VALUES ANALYSIS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. MISSING VALUES CHECK")
    print("=" * 50)
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": null_counts, "Percentage (%)": null_pct})
    if null_counts.sum() == 0:
        print("  [OK] No missing values found in any column (0 missing records, 100% complete).")
    else:
        print(missing_df[missing_df["Missing Count"] > 0])

    # --------------------------------------------------------------------------
    # 3. DUPLICATE ROWS CHECK
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. DUPLICATE ROWS CHECK")
    print("=" * 50)
    dup_count = df.duplicated().sum()
    print(f"  * Duplicate Rows Found : {dup_count} ({(dup_count / len(df)) * 100:.2f}%)")
    if dup_count == 0:
        print("  [OK] All records are distinct unique entries.")

    # --------------------------------------------------------------------------
    # 4. DATA TYPES & TEMPORAL RANGE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. DATA TYPES & TEMPORAL CONTINUITY")
    print("=" * 50)
    for col in df.columns:
        print(f"  * {col:<18} : {str(df[col].dtype):<10} | Sample: {df[col].iloc[0]}")
    
    dates = pd.to_datetime(df["date"])
    print(f"\n  * Date Range           : {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    print(f"  * Total Days Spanned   : {(dates.max() - dates.min()).days + 1} days")
    print(f"  * Unique Spatial Nodes : {len(df.groupby(['latitude', 'longitude']))} distinct station coordinates")

    # --------------------------------------------------------------------------
    # 5. STATISTICAL SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("5. STATISTICAL SUMMARY (NUMERICAL VARIABLES)")
    print("=" * 50)
    num_cols = ["temperature", "humidity", "pressure", "wind_speed", "rainfall", "elevation", "drainage_factor"]
    stats_df = df[num_cols].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats_df["skewness"] = df[num_cols].skew()
    stats_df["kurtosis"] = df[num_cols].kurtosis()
    print(stats_df.round(3).to_string())

    # --------------------------------------------------------------------------
    # 6. DISTRIBUTION OF RAINFALL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("6. DISTRIBUTION OF RAINFALL (ZERO-INFLATION & EXTREMES)")
    print("=" * 50)
    zero_rain = (df["rainfall"] == 0).sum()
    zero_rain_pct = (zero_rain / len(df)) * 100
    rain_days = df[df["rainfall"] > 0]["rainfall"]
    
    print(f"  * Zero-Rain Days (0.0 mm) : {zero_rain:,} ({zero_rain_pct:.2f}%)")
    print(f"  * Rainy Days (> 0.0 mm)   : {len(rain_days):,} ({100 - zero_rain_pct:.2f}%)")
    print(f"  * Mean on Rainy Days      : {rain_days.mean():.2f} mm")
    print(f"  * Median on Rainy Days    : {rain_days.median():.2f} mm")
    print(f"  * 90th Percentile Overall : {df['rainfall'].quantile(0.90):.2f} mm")
    print(f"  * 95th Percentile Overall : {df['rainfall'].quantile(0.95):.2f} mm")
    print(f"  * 99th Percentile Overall : {df['rainfall'].quantile(0.99):.2f} mm")
    print(f"  * Max Recorded Deluge     : {df['rainfall'].max():.2f} mm")

    # Plot 1: Rainfall Distribution
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1a. Overall Histogram with log-scale y
    sns.histplot(df["rainfall"], bins=50, kde=True, color="#0284c7", ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_title("Overall Rainfall Distribution (Log Count)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Rainfall (mm)")
    axes[0].set_ylabel("Log Frequency")

    # 1b. Non-zero Rainfall distribution (Log-transformed)
    log_rain = np.log1p(rain_days)
    sns.histplot(log_rain, bins=40, kde=True, color="#0d9488", ax=axes[1])
    axes[1].set_title("Non-Zero Rainfall: log(1 + mm)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("log(1 + Rainfall in mm)")
    axes[1].set_ylabel("Frequency")

    # 1c. Zero vs Rain Breakdown
    axes[2].pie(
        [zero_rain, len(rain_days)],
        labels=[f"Dry Days\n({zero_rain_pct:.1f}%)", f"Rainy Days\n({100-zero_rain_pct:.1f}%)"],
        autopct="%1.1f%%",
        colors=["#e2e8f0", "#38bdf8"],
        startangle=140,
        explode=(0, 0.08),
        wedgeprops={"edgecolor": "#64748b", "linewidth": 1}
    )
    axes[2].set_title("Dry vs Rainy Days Proportion", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "01_rainfall_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  [+] Saved plot: {fig1_path}")

    # --------------------------------------------------------------------------
    # 7. DISTRIBUTION OF HEAVY RAIN
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("7. DISTRIBUTION OF HEAVY RAIN (BINARY TARGET >= 64.5 mm)")
    print("=" * 50)
    hr_counts = df["heavy_rain"].value_counts()
    hr_pct = df["heavy_rain"].value_counts(normalize=True) * 100
    print(f"  * Class 0 [Normal / Moderate Rain (<64.5 mm)] : {hr_counts.get(0, 0):>6,} rows ({hr_pct.get(0, 0):>5.2f}%)")
    print(f"  * Class 1 [Heavy Rainfall Event  (>=64.5 mm)] : {hr_counts.get(1, 0):>6,} rows ({hr_pct.get(1, 0):>5.2f}%)")
    print(f"  * Class Imbalance Ratio                       : 1 : {hr_counts.get(0, 0) / hr_counts.get(1, 1):.1f}")

    # Plot 2: Heavy Rain Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["Normal / Light / Moderate (<64.5mm)", "Heavy Rain Event (>=64.5mm)"], hr_counts.values, color=["#38bdf8", "#ef4444"], edgecolor="#1e293b", linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,}\n({height/len(df)*100:.2f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count of Observations", fontsize=11)
    ax.set_title("Target Class Distribution: heavy_rain (IMD Standard >= 64.5 mm)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, len(df) * 1.12)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "02_heavy_rain_target_distribution.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  [+] Saved plot: {fig2_path}")

    # --------------------------------------------------------------------------
    # 8. CORRELATION BETWEEN NUMERICAL VARIABLES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("8. CORRELATION ANALYSIS (PEARSON & SPEARMAN)")
    print("=" * 50)
    corr_cols = ["temperature", "humidity", "pressure", "wind_speed", "rainfall", "elevation", "drainage_factor", "heavy_rain"]
    corr_matrix = df[corr_cols].corr()
    
    print("\n--- Pearson Correlation with 'rainfall' ---")
    rain_corr = corr_matrix["rainfall"].sort_values(ascending=False)
    for feat, val in rain_corr.items():
        if feat != "rainfall":
            print(f"  * {feat:<18} : {val:>+7.4f}")

    print("\n--- Pearson Correlation with 'heavy_rain' ---")
    hr_corr = corr_matrix["heavy_rain"].sort_values(ascending=False)
    for feat, val in hr_corr.items():
        if feat != "heavy_rain":
            print(f"  * {feat:<18} : {val:>+7.4f}")

    # Plot 3: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Multi-Variable Pearson Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "03_feature_correlation_heatmap.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  [+] Saved plot: {fig3_path}")

    # Plot 4: Atmospheric Drivers (Humidity vs Pressure vs Rainfall)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["pressure"],
        df["humidity"],
        c=df["rainfall"],
        cmap="viridis",
        s=18,
        alpha=0.6,
        edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Rainfall Amount (mm)", fontsize=11)
    ax.set_xlabel("Surface Atmospheric Pressure (hPa)", fontsize=11)
    ax.set_ylabel("Relative Humidity (%)", fontsize=11)
    ax.set_title("Atmospheric Trigger Convergence: Humidity vs Pressure vs Rainfall Deluge", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "04_rainfall_atmospheric_interaction.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  [+] Saved plot: {fig4_path}")

    # --------------------------------------------------------------------------
    # 9. OUTLIER DETECTION (IQR METHOD & BOXPLOTS)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("9. OUTLIER DETECTION (INTERQUARTILE RANGE - IQR METHOD)")
    print("=" * 50)
    
    outlier_summary = []
    for col in num_cols:
        q25 = df[col].quantile(0.25)
        q75 = df[col].quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        n_outliers = len(outliers)
        pct_outliers = (n_outliers / len(df)) * 100
        
        outlier_summary.append({
            "Feature": col,
            "Q25": round(q25, 2),
            "Q75": round(q75, 2),
            "IQR": round(iqr, 2),
            "Lower Bound": round(lower_bound, 2),
            "Upper Bound": round(upper_bound, 2),
            "Outlier Count": n_outliers,
            "Outlier (%)": round(pct_outliers, 2)
        })
        
        print(f"  * {col:<18} : {n_outliers:>5,} outliers ({pct_outliers:>5.2f}%) | Bounds: [{lower_bound:>8.2f}, {upper_bound:>8.2f}]")

    print("\n--- Outlier Hydrological Interpretation ---")
    print("  Note: Rainfall outliers (values > 8.66 mm) represent authentic heavy/extreme downpours")
    print("  and cloudburst events that form the critical hazard signal for the Jal Sanketh system.")

    # Plot 5: Boxplots for all numerical variables
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    
    palette = sns.color_palette("tab10", len(num_cols))
    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col], ax=axes[i], color=palette[i], fliersize=3)
        axes[i].set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
        axes[i].set_ylabel(col, fontsize=10)
        
    # Hide unused 8th subplot
    axes[7].set_visible(False)
    
    plt.suptitle("Outlier & Distribution Boxplots across Numerical Features", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "05_feature_boxplots_outliers.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"  [+] Saved plot: {fig5_path}")

    print("\n" + "=" * 80)
    print("  [OK] EDA Completed successfully. 5 figure files saved to reports/figures/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    raw_data_file = os.path.join("data", "raw", "weather_demo.csv")
    output_fig_dir = os.path.join("reports", "figures")
    run_eda(data_path=raw_data_file, figures_dir=output_fig_dir)

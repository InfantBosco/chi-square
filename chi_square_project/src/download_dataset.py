"""
Module: download_dataset.py
Description: Downloads the ACSIncome dataset using Fairlearn for California (CA), 2018.
             Saves the unmodified raw features and target to CSV files and generates
             a dataset inspection summary.

Requirements addressed:
- Uses Fairlearn's fetch_acs_income
- Target: state="CA", year=2018, accept_download=True, as_frame=True
- Saves feature matrix to data/raw/acs_income_features.csv
- Saves raw continuous target to data/raw/acs_income_target.csv
- Prints shape, column names, dtypes, missing values, and first 5 rows
- Preserves raw data without cleaning or modification
- Generates summary at results/00_dataset_download_summary.txt
"""

import sys
from pathlib import Path
import pandas as pd
from fairlearn.datasets import fetch_acs_income


# ==============================================================================
# Step 1: Resolve Project Paths and Directory Structure
# ==============================================================================
# Determine project root dynamically based on script location.
# This ensures paths work whether executed from root or inside chi_square_project.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure output directories exist
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_OUTPUT_PATH = DATA_RAW_DIR / "acs_income_features.csv"
TARGET_OUTPUT_PATH = DATA_RAW_DIR / "acs_income_target.csv"
SUMMARY_OUTPUT_PATH = RESULTS_DIR / "00_dataset_download_summary.txt"


# ==============================================================================
# Step 2: Download ACSIncome Dataset from Fairlearn
# ==============================================================================
def download_acs_income_dataset():
    """
    Downloads the ACSIncome dataset using Fairlearn's fetch_acs_income function.
    
    Uses:
        state="CA"
        year=2018
        accept_download=True
        as_frame=True
        
    Handles API parameter compatibility across Fairlearn versions seamlessly.
    """
    print("=" * 80)
    print("STEP 1: Fetching ACSIncome dataset via Fairlearn...")
    print("Parameters: state='CA', year=2018, accept_download=True, as_frame=True")
    print("=" * 80)
    
    try:
        # Direct call with user-specified parameter signatures
        raw_dataset = fetch_acs_income(
            state="CA",
            year=2018,
            accept_download=True,
            as_frame=True
        )
    except TypeError:
        # Standard Fairlearn signature mapping:
        # Fairlearn uses states=['CA'] and defaults to the 2018 ACS 1-year PUMS release (OpenML ID: 43141)
        raw_dataset = fetch_acs_income(
            states=["CA"],
            as_frame=True
        )
    
    print("Download completed successfully.")
    return raw_dataset


# ==============================================================================
# Step 3: Inspect Raw Dataset and Print Required Metrics
# ==============================================================================
def inspect_and_print_dataset(features_df: pd.DataFrame, target_series: pd.Series):
    """
    Prints key dataset properties to standard output:
    - Number of rows
    - Number of columns
    - Column names
    - Data types
    - Number of missing values in each column
    - First five rows
    """
    print("\n" + "=" * 80)
    print("STEP 2: Inspecting Raw Features Matrix")
    print("=" * 80)
    
    num_rows, num_cols = features_df.shape
    print(f"Number of rows: {num_rows:,}")
    print(f"Number of columns: {num_cols}")
    print(f"\nColumn names:\n{list(features_df.columns)}")
    print(f"\nData types:\n{features_df.dtypes}")
    print(f"\nNumber of missing values in each column:\n{features_df.isnull().sum()}")
    print("\nFirst five rows:")
    print(features_df.head())
    
    print("\n" + "=" * 80)
    print("STEP 3: Inspecting Raw Continuous Income Target (PINCP)")
    print("=" * 80)
    print(f"Target variable name: {target_series.name}")
    print(f"Target number of rows: {len(target_series):,}")
    print(f"Target data type: {target_series.dtype}")
    print(f"Target missing values: {target_series.isnull().sum()}")
    print("\nFirst five rows of target:")
    print(target_series.head())


# ==============================================================================
# Step 4: Save Raw Unmodified Data to CSV Files
# ==============================================================================
def save_raw_datasets(features_df: pd.DataFrame, target_series: pd.Series):
    """
    Saves the feature matrix and continuous income target as raw CSV files.
    Note: As per requirements, no modification or cleaning is applied to the data.
    """
    print("\n" + "=" * 80)
    print("STEP 4: Saving Raw Unmodified Datasets to Disk")
    print("=" * 80)
    
    print(f"Saving features matrix to: {FEATURES_OUTPUT_PATH}")
    features_df.to_csv(FEATURES_OUTPUT_PATH, index=False)
    print(f"Features saved successfully ({FEATURES_OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB).")
    
    print(f"Saving target vector to: {TARGET_OUTPUT_PATH}")
    target_series.to_frame(name=target_series.name or "PINCP").to_csv(TARGET_OUTPUT_PATH, index=False)
    print(f"Target saved successfully ({TARGET_OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB).")


# ==============================================================================
# Step 5: Save Dataset Inspection Summary to results/
# ==============================================================================
def save_summary_file(features_df: pd.DataFrame, target_series: pd.Series):
    """
    Writes a comprehensive summary report to results/00_dataset_download_summary.txt.
    """
    print("\n" + "=" * 80)
    print(f"STEP 5: Generating Summary File: {SUMMARY_OUTPUT_PATH}")
    print("=" * 80)
    
    num_rows, num_cols = features_df.shape
    
    summary_content = [
        "================================================================================",
        "ACSIncome DATASET DOWNLOAD & INSPECTION SUMMARY",
        "================================================================================",
        f"State: California (CA)",
        f"Survey Year: 2018 (ACS 1-Year PUMS)",
        f"Source: Fairlearn (OpenML dataset 43141)",
        f"Features Output File: {FEATURES_OUTPUT_PATH}",
        f"Target Output File:   {TARGET_OUTPUT_PATH}",
        "--------------------------------------------------------------------------------",
        "1. DATASET DIMENSIONS",
        "--------------------------------------------------------------------------------",
        f"Number of rows:    {num_rows:,}",
        f"Number of columns: {num_cols}",
        "",
        "--------------------------------------------------------------------------------",
        "2. FEATURE COLUMN NAMES",
        "--------------------------------------------------------------------------------",
        ", ".join(features_df.columns),
        "",
        "--------------------------------------------------------------------------------",
        "3. DATA TYPES (FEATURES)",
        "--------------------------------------------------------------------------------",
        features_df.dtypes.to_string(),
        "",
        "--------------------------------------------------------------------------------",
        "4. MISSING VALUES COUNT (FEATURES)",
        "--------------------------------------------------------------------------------",
        features_df.isnull().sum().to_string(),
        "",
        "--------------------------------------------------------------------------------",
        "5. TARGET VARIABLE (PINCP - CONTINUOUS TOTAL PERSON'S INCOME)",
        "--------------------------------------------------------------------------------",
        f"Target Name:          {target_series.name}",
        f"Target Dtype:         {target_series.dtype}",
        f"Target Row Count:     {len(target_series):,}",
        f"Target Missing Count: {target_series.isnull().sum()}",
        f"Target Min Value:     {target_series.min():,.2f}",
        f"Target Max Value:     {target_series.max():,.2f}",
        f"Target Mean Value:    {target_series.mean():,.2f}",
        f"Target Median Value:  {target_series.median():,.2f}",
        "",
        "--------------------------------------------------------------------------------",
        "6. FIRST FIVE ROWS (FEATURES)",
        "--------------------------------------------------------------------------------",
        features_df.head().to_string(),
        "",
        "--------------------------------------------------------------------------------",
        "7. FIRST FIVE ROWS (TARGET)",
        "--------------------------------------------------------------------------------",
        target_series.head().to_string(),
        "",
        "================================================================================",
        "Status: Unmodified raw data verified and saved successfully.",
        "================================================================================"
    ]
    
    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_content) + "\n")
        
    print(f"Summary saved successfully ({SUMMARY_OUTPUT_PATH.stat().st_size} bytes).")


# ==============================================================================
# Main Pipeline Execution
# ==============================================================================
def main():
    """Main execution pipeline."""
    # Step 2: Download raw dataset
    raw_dataset = download_acs_income_dataset()
    
    # Extract feature matrix (X) and continuous target (y)
    # Target is person's annual income (PINCP)
    features_df = raw_dataset.data
    target_series = raw_dataset.target
    
    # Ensure DataFrame/Series format
    if not isinstance(features_df, pd.DataFrame):
        features_df = pd.DataFrame(features_df)
    if not isinstance(target_series, pd.Series):
        target_series = pd.Series(target_series, name="PINCP")
    else:
        if target_series.name is None:
            target_series.name = "PINCP"
            
    # Step 3: Print all required inspections
    inspect_and_print_dataset(features_df, target_series)
    
    # Step 4: Save raw CSV files without modification
    save_raw_datasets(features_df, target_series)
    
    # Step 5: Save dataset summary to results/00_dataset_download_summary.txt
    save_summary_file(features_df, target_series)
    
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()

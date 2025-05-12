#!/usr/bin/env python3
"""
Read Parquet files from the project.

This module provides utilities to read parquet files into pandas DataFrames.
"""

import os
import pandas as pd
import glob
import sys
import subprocess
from typing import Dict, List, Optional, Union
import argparse


def check_dependencies():
    """
    Check if required dependencies are installed and install them if missing.
    
    Returns:
        bool: True if dependencies are available, False otherwise
    """
    try:
        import pyarrow
        print(f"pyarrow version {pyarrow.__version__} is already installed.")
        return True
    except ImportError:
        print("pyarrow is not installed. Attempting to install...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow>=15.0.0"])
            print("pyarrow has been successfully installed.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install pyarrow: {e}")
            print("\nPlease manually install pyarrow with:")
            print("    pip install pyarrow>=15.0.0")
            return False


def read_parquet_file(file_path: str) -> pd.DataFrame:
    """
    Read a single parquet file into a pandas DataFrame.
    
    Args:
        file_path: Path to the parquet file
        
    Returns:
        DataFrame containing the parquet file data
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: Other errors that might occur during reading
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Explicitly use pyarrow engine
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"Successfully read parquet file: {file_path}")
        print(f"DataFrame shape: {df.shape}")
        return df
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Error reading parquet file {file_path}: {e}")
        raise


def read_parquet_directory(directory_path: str, pattern: str = "*.parquet") -> Dict[str, pd.DataFrame]:
    """
    Read all parquet files in a directory matching the given pattern.
    
    Args:
        directory_path: Path to the directory containing parquet files
        pattern: Glob pattern to match files (default: "*.parquet")
        
    Returns:
        Dictionary mapping file names to DataFrames
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return {}
    
    parquet_files = glob.glob(os.path.join(directory_path, pattern))
    
    if not parquet_files:
        print(f"No parquet files found in {directory_path} matching pattern {pattern}")
        return {}
    
    dataframes = {}
    for file_path in parquet_files:
        try:
            file_name = os.path.basename(file_path)
            dataframes[file_name] = read_parquet_file(file_path)
        except Exception as e:
            print(f"Skipping file {file_path} due to error: {e}")
    
    return dataframes


def get_dataset_sample(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return a sample of rows from a DataFrame.
    
    Args:
        df: Input DataFrame
        n: Number of rows to sample (default: 5)
        
    Returns:
        DataFrame containing sample rows
    """
    return df.head(n)


def show_column_info(df: pd.DataFrame) -> None:
    """
    Print information about DataFrame columns.
    
    Args:
        df: Input DataFrame
    """
    print("\nColumn Information:")
    print("-" * 50)
    for col in df.columns:
        print(f"Column: {col}")
        print(f"  Type: {df[col].dtype}")
        try:
            unique_count = df[col].nunique()
            print(f"  Unique values: {unique_count}")
        except TypeError:
            print(f"  Unique values: Cannot compute (unhashable type)")
        
        try:
            # Get the first 3 values, convert unhashable types to strings
            sample_values = []
            for i, val in enumerate(df[col].head(3)):
                if i >= 3:
                    break
                try:
                    # Try to add as is first
                    sample_values.append(val)
                except (TypeError, ValueError):
                    # If unhashable, convert to string representation
                    sample_values.append(str(val))
            print(f"  Sample values: {sample_values}")
        except Exception as e:
            print(f"  Sample values: Error retrieving sample ({str(e)})")
        
        print("-" * 50)


if __name__ == "__main__":
    # First check dependencies before attempting to read files
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='math-base')
    args = parser.parse_args()

    if not check_dependencies():
        print("Required dependencies are missing. Please install them before running this script.")
        sys.exit(1)
    
    # Example usage
    data_dir = os.path.join("./data", args.dataset_name)
    
    print("Reading individual parquet files...")
    try:
        train_df = read_parquet_file(os.path.join(data_dir, "train.parquet"))
        print("\nTrain data sample:")
        print(get_dataset_sample(train_df))
        show_column_info(train_df)

        print(f"Column names: {train_df.columns}")
        print(f"Example row: {train_df.iloc[0]}")
        print(f"Example row: {train_df.iloc[0]['data_source']}")
        print(f"Example row: {train_df.iloc[0]['prompt']}")
        print(f"Example row: {train_df.iloc[0]['ability']}")
        print(f"Example row: {train_df.iloc[0]['reward_model']}")
        print(f"Example row: {train_df.iloc[0]['extra_info']}")
    except Exception as e:
        print(f"Error processing train data: {e}")

    try:
        test_df = read_parquet_file(os.path.join(data_dir, "test.parquet"))
        print("\nTest data sample:")
        print(get_dataset_sample(test_df))
        show_column_info(test_df)
        print(f"Column names: {test_df.columns}")
        print(f"Example row: {test_df.iloc[0]}")
        print(f"Example row: {test_df.iloc[0]['data_source']}")
        print(f"Example row: {test_df.iloc[0]['prompt']}")
        print(f"Example row: {test_df.iloc[0]['ability']}")
        print(f"Example row: {test_df.iloc[0]['reward_model']}")
        print(f"Example row: {test_df.iloc[0]['extra_info']}")
    except Exception as e:
        print(f"Error processing test data: {e}")
    import pdb; pdb.set_trace()

    
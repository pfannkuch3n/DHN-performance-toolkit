"""
Decentral Pump Power Analysis
============================
Calculates energy consumption of decentralized heating pumps.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from typing import Callable, Optional

def calculate_decentral_pump_energy(df: pd.DataFrame, output_dir: str, store_result = True) -> pd.DataFrame:
    """
    Calculate the energy consumption of decentralized pumps from the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pump power data.
    output_dir : str
        Directory to save the results and plots.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the energy consumption of decentralized pumps.
    """
    time_diffs = df.index.to_series().diff().dt.total_seconds() / 3600
    time_diffs.iloc[0] = 0
    energy_df = df.multiply(time_diffs, axis=0)* 1e-3  # Convert from W to kW
    energy_df = transform_column_names(energy_df, suffix=' [kWh]')
    energy_cum_df = energy_df.cumsum()
    energy_cum_total_df = energy_cum_df.sum(axis=1).to_frame(name='Cumulative Energy [kWh]')
    plot_energy_consumption(energy_cum_total_df, output_dir, label="decentralApproach")
    results_dict = {
        "Pump Power": df,
        "Pump Energy": energy_df,
        "Pump Energy cumulated": energy_cum_df,
        "Pump Energy cumulated total": energy_cum_total_df
    }
    combined_df = combine_dataframes(results_dict)
    if store_result:
        res = {"Pump Power calculation": combined_df}
        store_results(output_dir, res, name="decentral_pump_power")
    return combined_df
    
def store_results(output_dir, results, name="results"):
    if os.path.exists(output_dir):
        try:
            with pd.ExcelWriter(f"{output_dir}/{name}.xlsx") as writer:
                for key in results.keys():
                    results[key].to_excel(writer, sheet_name=key)
        except PermissionError:
            print("Please close the Excel file before running the script again.")
    else:
        print(f"Output directory '{output_dir}' does not exist. Results will not be stored.")


def plot_energy_consumption(df, output_dir,label =""):
    df.plot()
    plt.title(f'Cumulative Pump Power Consumption {label}')
    plt.xlabel('Time')
    plt.ylabel('Power Consumption (kWh)')
    plt.savefig(f"{output_dir}/Pump_power_consumption_{label}.png")
    plt.close()


def combine_dataframes(results_dict):
    combined_df = pd.DataFrame()

    for key, df in results_dict.items():
        # Anpassen der Spaltennamen, um den Ursprung der Daten widerzuspiegeln
        renamed_columns = {col: f"{key}_{col}" for col in df.columns}
        df = df.rename(columns=renamed_columns)

        # Zusammenführen der DataFrames
        combined_df = pd.concat([combined_df, df], axis=1)

    return combined_df


def transform_column_names(
    df: pd.DataFrame,
    pattern: str = r'T([^.]+)'  ,
    prefix: str = '',
    suffix: str = '',
    transform_func: Optional[Callable[[str], str]] = None,
    inplace: bool = False,
    keep_unmatched: bool = True
) -> pd.DataFrame:
    """
    Transforms DataFrame column names by extracting patterns and applying customizations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with column names to transform
    pattern : str, default r'T([^.]+)'
        Regular expression pattern to match in column names
    prefix : str, default ''
        String to add before the extracted pattern
    suffix : str, default ''
        String to add after the extracted pattern
    transform_func : Optional[Callable[[str], str]], default None
        Optional function to further transform the extracted pattern
        Function should take a string and return a string
    inplace : bool, default False
        If True, modifies the original DataFrame, otherwise returns a copy
    keep_unmatched : bool, default True
        If True, keeps columns that don't match the pattern unchanged
        If False, drops columns that don't match the pattern

    Returns
    -------
    pd.DataFrame
        DataFrame with transformed column names

    Examples
    --------
    >>> df = pd.DataFrame(columns=['networkModel.demandT107.heatPump.P',
    ...                            'networkModel.demandT205.heatPump.P'])
    >>> transform_column_names(df, suffix='[W]')
    # Returns DataFrame with columns renamed to ['T107[W]', 'T205[W]']
    """
    compiled_pattern = re.compile(pattern)

    name_mapping = {}
    columns_to_keep = []

    for col in df.columns:
        match = compiled_pattern.search(col)
        if match:
            extracted_value = match.group(1)

            if transform_func:
                extracted_value = transform_func(extracted_value)

            new_name = f"{prefix}{extracted_value}{suffix}"

            name_mapping[col] = new_name
            columns_to_keep.append(col)
        elif keep_unmatched:
            name_mapping[col] = col
            columns_to_keep.append(col)

    if inplace:
        result_df = df
        if not keep_unmatched:
            result_df = result_df[columns_to_keep]
        result_df.columns = [name_mapping.get(col, col) for col in result_df.columns]
    else:
        result_df = df.copy()
        if not keep_unmatched:
            result_df = result_df[columns_to_keep]
        result_df.columns = [name_mapping.get(col, col) for col in result_df.columns]

    return result_df
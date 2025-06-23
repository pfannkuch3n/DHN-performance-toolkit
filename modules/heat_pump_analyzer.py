import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Tuple, Any
import logging
import warnings


from .utils import set_up_terminal_logger

def calculate_heat_pump_cop(
    df_thermal_output: pd.DataFrame,
    df_hp_electrical: pd.DataFrame,
    df_pump_electrical: pd.DataFrame,
    include_pump_power: bool = True,
    min_power_threshold: float = 0.1,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Calculate Coefficient of Performance (COP) for heat pumps.
    
    The COP is calculated as:
    COP = Thermal Output / Total Electrical Input
    
    Where Total Electrical Input = Heat Pump Power + Pump Power (optional)
    
    Parameters
    ----------
    df_thermal_output : pd.DataFrame
        DataFrame containing thermal output power (Q_con) for each heat pump.
        Columns represent individual heat pumps, rows represent time steps.
        Units should be consistent with electrical power (typically kW).
        
    df_hp_electrical : pd.DataFrame
        DataFrame containing electrical power consumption of heat pumps.
        Must have same structure as df_thermal_output.
        
    df_pump_electrical : pd.DataFrame
        DataFrame containing electrical power consumption of circulation pumps.
        Must have same structure as df_thermal_output.
        
    include_pump_power : bool, default True
        Whether to include pump power in the electrical input calculation.
        If False, only heat pump electrical power is considered.
        
    min_power_threshold : float, default 0.1
        Minimum electrical power threshold (in same units as input data).
        Below this threshold, the heat pump is considered "off" and COP is set to NaN.
        This prevents artificially high COPs during standby periods.
        
    logger : logging.Logger, optional
        Logger instance for warnings and information. If None, a default logger is created.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - DataFrame containing calculated COP values with same structure as input DataFrames
        - Dictionary containing calculation statistics and warnings:
          * 'total_calculations': Total number of COP calculations performed
          * 'valid_calculations': Number of valid (non-NaN) COP values
          * 'zero_power_warnings': Number of cases where thermal output > 0 but electrical power ≈ 0
          * 'division_by_zero_cases': Number of division by zero cases handled
          * 'mean_cop_by_building': Dictionary of mean COP values per building
          * 'overall_mean_cop': Mean COP across all valid calculations
    
    Warns
    -----
    UserWarning
        - When thermal output is positive but electrical power is near zero
        - When input DataFrames have mismatched dimensions
        - When a significant number of calculations result in NaN values
    
    Notes
    -----
    - NaN values in input data are preserved and propagate to the output
    - Zero or negative thermal output results in NaN COP (heat pumps should only produce heat)
    - Very low electrical power (below threshold) results in NaN COP to avoid unrealistic values
    - The function handles edge cases gracefully and provides detailed logging
    
    Examples
    --------
    >>> # Basic COP calculation including pump power
    >>> cop_df, stats = calculate_heat_pump_cop(
    ...     df_thermal_output=df_hp_q_con,
    ...     df_hp_electrical=df_hp_power,
    ...     df_pump_electrical=df_pump_power
    ... )
    
    >>> # COP calculation without pump power, with custom threshold
    >>> cop_df, stats = calculate_heat_pump_cop(
    ...     df_thermal_output=df_hp_q_con,
    ...     df_hp_electrical=df_hp_power,
    ...     df_pump_electrical=df_pump_power,
    ...     include_pump_power=False,
    ...     min_power_threshold=0.05
    ... )
    
    >>> print(f"Overall mean COP: {stats['overall_mean_cop']:.2f}")
    >>> print(f"Valid calculations: {stats['valid_calculations']}/{stats['total_calculations']}")
    """
    
    # Set up logger if not provided
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.calculate_heat_pump_cop")
    
    # Input validation
    logger.info("Starting heat pump COP calculation")
    
    if not all(isinstance(df, pd.DataFrame) for df in [df_thermal_output, df_hp_electrical, df_pump_electrical]):
        raise TypeError("All input arguments must be pandas DataFrames")
    
    # Check DataFrame dimensions
    if not (df_thermal_output.shape == df_hp_electrical.shape == df_pump_electrical.shape):
        error_msg = (f"DataFrame dimensions mismatch: "
                    f"thermal_output: {df_thermal_output.shape}, "
                    f"hp_electrical: {df_hp_electrical.shape}, "
                    f"pump_electrical: {df_pump_electrical.shape}")
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check column alignment
    if not (df_thermal_output.columns.equals(df_hp_electrical.columns) and 
            df_thermal_output.columns.equals(df_pump_electrical.columns)):
        logger.warning("Column names are not identical between DataFrames. Proceeding with position-based alignment.")
    
    # Check index alignment
    if not (df_thermal_output.index.equals(df_hp_electrical.index) and 
            df_thermal_output.index.equals(df_pump_electrical.index)):
        logger.warning("Index values are not identical between DataFrames. Proceeding with position-based alignment.")
    
    logger.info(f"Processing data with shape: {df_thermal_output.shape}")
    logger.info(f"Include pump power: {include_pump_power}")
    logger.info(f"Minimum power threshold: {min_power_threshold}")
    
    # Calculate total electrical power
    if include_pump_power:
        df_total_electrical = df_hp_electrical + df_pump_electrical
        logger.info("Using heat pump + pump electrical power for COP calculation")
    else:
        df_total_electrical = df_hp_electrical.copy()
        logger.info("Using only heat pump electrical power for COP calculation")
    
    # Initialize statistics tracking
    stats = {
        'total_calculations': df_thermal_output.size,
        'valid_calculations': 0,
        'zero_power_warnings': 0,
        'division_by_zero_cases': 0,
        'mean_cop_by_building': {},
        'overall_mean_cop': np.nan
    }
    
    # Initialize result DataFrame with same structure
    df_cop = pd.DataFrame(
        index=df_thermal_output.index,
        columns=df_thermal_output.columns,
        dtype=float
    )
    
    # Perform COP calculation with comprehensive error handling
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        
        # Element-wise calculation with conditions
        for col in df_thermal_output.columns:
            for idx in df_thermal_output.index:
                
                thermal = df_thermal_output.loc[idx, col]
                electrical = df_total_electrical.loc[idx, col]
                
                # Handle NaN values
                if pd.isna(thermal) or pd.isna(electrical):
                    df_cop.loc[idx, col] = np.nan
                    continue
                
                # Check for negative thermal output (unphysical)
                if thermal <= 0:
                    df_cop.loc[idx, col] = np.nan
                    continue
                
                # Check for problematic cases: thermal output but no electrical power
                if thermal > 0 and abs(electrical) < min_power_threshold:
                    stats['zero_power_warnings'] += 1
                    df_cop.loc[idx, col] = np.nan
                    
                    if stats['zero_power_warnings'] <= 5:  # Limit warning spam
                        logger.warning(
                            f"Building {col} at {idx}: Thermal output ({thermal:.3f}) > 0 "
                            f"but electrical power ({electrical:.3f}) < threshold ({min_power_threshold}). "
                            f"Setting COP to NaN."
                        )
                    continue
                
                # Check for division by zero or very small denominators
                if abs(electrical) < 1e-10:
                    stats['division_by_zero_cases'] += 1
                    df_cop.loc[idx, col] = np.nan
                    continue
                
                # Calculate COP
                cop_value = thermal / electrical
                
                # Sanity check: COP should be reasonable for heat pumps (typically 1.5-8.0)
                if cop_value > 15:
                    logger.warning(
                        f"Unusually high COP ({cop_value:.2f}) for building {col} at {idx}. "
                        f"Thermal: {thermal:.3f}, Electrical: {electrical:.3f}"
                    )
                elif cop_value < 1:
                    logger.warning(
                        f"Low COP ({cop_value:.2f}) for building {col} at {idx}. "
                        f"Thermal: {thermal:.3f}, Electrical: {electrical:.3f}"
                    )
                
                df_cop.loc[idx, col] = cop_value
                stats['valid_calculations'] += 1
    
    # Calculate statistics
    logger.info("Calculating COP statistics")
    
    # Per-building mean COP
    for col in df_cop.columns:
        valid_cops = df_cop[col].dropna()
        if len(valid_cops) > 0:
            stats['mean_cop_by_building'][str(col)] = float(valid_cops.mean())
        else:
            stats['mean_cop_by_building'][str(col)] = np.nan
    
    # Overall mean COP
    all_valid_cops = df_cop.stack().dropna()
    if len(all_valid_cops) > 0:
        stats['overall_mean_cop'] = float(all_valid_cops.mean())
    
    # Log summary statistics
    logger.info(f"COP calculation completed:")
    logger.info(f"  Total calculations: {stats['total_calculations']}")
    logger.info(f"  Valid calculations: {stats['valid_calculations']}")
    logger.info(f"  Success rate: {stats['valid_calculations']/stats['total_calculations']*100:.1f}%")
    logger.info(f"  Overall mean COP: {stats['overall_mean_cop']:.2f}")
    
    if stats['zero_power_warnings'] > 0:
        logger.warning(f"Found {stats['zero_power_warnings']} cases with thermal output but insufficient electrical power")
    
    if stats['division_by_zero_cases'] > 0:
        logger.warning(f"Handled {stats['division_by_zero_cases']} division by zero cases")
    
    # Issue warnings for data quality issues
    if stats['valid_calculations'] / stats['total_calculations'] < 0.5:
        warnings.warn(
            f"Less than 50% of COP calculations were valid ({stats['valid_calculations']}/{stats['total_calculations']}). "
            "Check input data quality and consider adjusting min_power_threshold.",
            UserWarning
        )
    
    return df_cop, stats

def calculate_energy_from_power(
    df_power: pd.DataFrame,
    time_interval_hours: float = 0.25  # 15 minutes = 0.25 hours
) -> pd.DataFrame:
    """
    Convert power [kW] to energy [kWh] using time intervals.
    
    Parameters
    ----------
    df_power : pd.DataFrame
        Power data in kW
    time_interval_hours : float, default 0.25
        Time interval between measurements in hours (15 min = 0.25h)
        
    Returns
    -------
    pd.DataFrame
        Energy data in kWh (same structure as input)
    """
    # For constant power assumption: Energy = Power × Time
    return df_power * time_interval_hours


def analyze_monthly_energy_cop(
    df_thermal_power: pd.DataFrame,
    df_hp_electrical_power: pd.DataFrame,
    df_pump_electrical_power: pd.DataFrame,
    time_interval_hours: float = 0.25,
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate monthly energy consumption and COP based on actual energy balances.
    
    This is much more reliable than averaging power values, as it represents
    the true energy efficiency over each month.
    
    Parameters
    ----------
    df_thermal_power : pd.DataFrame
        Thermal power output [kW]
    df_hp_electrical_power : pd.DataFrame
        Heat pump electrical power [kW]
    df_pump_electrical_power : pd.DataFrame
        Pump electrical power [kW]
    time_interval_hours : float, default 0.25
        Time interval between measurements (15 min = 0.25h)
    save_path : str, optional
        Path to save results CSV
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        - Monthly energy and COP summary DataFrame
        - Dictionary with detailed statistics
    """
    
    print(f"Converting power to energy using {time_interval_hours}h intervals...")
    
    # Convert power to energy
    df_thermal_energy = calculate_energy_from_power(df_thermal_power, time_interval_hours)
    df_hp_energy = calculate_energy_from_power(df_hp_electrical_power, time_interval_hours)
    df_pump_energy = calculate_energy_from_power(df_pump_electrical_power, time_interval_hours)
    df_total_electrical_energy = df_hp_energy + df_pump_energy
    
    # Calculate monthly totals
    monthly_results = []
    
    for month in range(1, 13):
        month_mask = df_thermal_energy.index.month == month
        
        # Sum all energy across all buildings for this month
        thermal_energy_month = df_thermal_energy[month_mask].sum().sum()  # Total kWh
        hp_energy_month = df_hp_energy[month_mask].sum().sum()  # Total kWh
        pump_energy_month = df_pump_energy[month_mask].sum().sum()  # Total kWh
        total_electrical_month = df_total_electrical_energy[month_mask].sum().sum()  # Total kWh
        
        # Calculate energy-based COP
        if total_electrical_month > 0:
            energy_cop = thermal_energy_month / total_electrical_month
        else:
            energy_cop = np.nan
            
        if hp_energy_month > 0:
            energy_cop_hp_only = thermal_energy_month / hp_energy_month
        else:
            energy_cop_hp_only = np.nan
        
        # Count active hours (where thermal energy was produced)
        active_periods = (df_thermal_energy[month_mask] > 0).sum().sum()
        total_periods = df_thermal_energy[month_mask].size
        
        monthly_results.append({
            'Month': month,
            'Month_Name': pd.Timestamp(f'2024-{month:02d}-01').strftime('%B'),
            'Thermal_Energy_MWh': thermal_energy_month / 1000,  # Convert to MWh
            'HP_Electrical_MWh': hp_energy_month / 1000,
            'Pump_Electrical_MWh': pump_energy_month / 1000,
            'Total_Electrical_MWh': total_electrical_month / 1000,
            'COP_Including_Pump': energy_cop,
            'COP_HP_Only': energy_cop_hp_only,
            'Active_Periods': active_periods,
            'Total_Periods': total_periods,
            'Activity_Rate': active_periods / total_periods * 100,
            'Pump_Share_Percent': (pump_energy_month / total_electrical_month * 100) if total_electrical_month > 0 else 0
        })
    
    df_monthly = pd.DataFrame(monthly_results)
    
    # Calculate annual totals
    annual_stats = {
        'annual_thermal_MWh': df_monthly['Thermal_Energy_MWh'].sum(),
        'annual_electrical_MWh': df_monthly['Total_Electrical_MWh'].sum(),
        'annual_cop': df_monthly['Thermal_Energy_MWh'].sum() / df_monthly['Total_Electrical_MWh'].sum(),
        'annual_cop_hp_only': df_monthly['Thermal_Energy_MWh'].sum() / df_monthly['HP_Electrical_MWh'].sum(),
        'winter_cop': df_monthly[df_monthly['Month'].isin([12, 1, 2])]['COP_Including_Pump'].mean(),
        'summer_cop': df_monthly[df_monthly['Month'].isin([6, 7, 8])]['COP_Including_Pump'].mean(),
        'heating_season_energy': df_monthly[df_monthly['Month'].isin([10, 11, 12, 1, 2, 3, 4])]['Thermal_Energy_MWh'].sum(),
        'non_heating_season_energy': df_monthly[df_monthly['Month'].isin([5, 6, 7, 8, 9])]['Thermal_Energy_MWh'].sum()
    }
    
    # Print summary
    print("\n" + "="*80)
    print("MONTHLY ENERGY-BASED COP ANALYSIS")
    print("="*80)
    print(f"{'Month':<10} {'Thermal':<10} {'Electrical':<12} {'COP':<6} {'COP(HP)':<8} {'Activity':<8}")
    print(f"{'':10} {'[MWh]':<10} {'[MWh]':<12} {'':6} {'only':<8} {'[%]':<8}")
    print("-" * 80)
    
    for _, row in df_monthly.iterrows():
        print(f"{row['Month_Name']:<10} "
              f"{row['Thermal_Energy_MWh']:<10.1f} "
              f"{row['Total_Electrical_MWh']:<12.1f} "
              f"{row['COP_Including_Pump']:<6.2f} "
              f"{row['COP_HP_Only']:<8.2f} "
              f"{row['Activity_Rate']:<8.1f}")
    
    print("-" * 80)
    print(f"{'ANNUAL':<10} "
          f"{annual_stats['annual_thermal_MWh']:<10.1f} "
          f"{annual_stats['annual_electrical_MWh']:<12.1f} "
          f"{annual_stats['annual_cop']:<6.2f} "
          f"{annual_stats['annual_cop_hp_only']:<8.2f}")
    
    print(f"\nSeasonal Comparison:")
    print(f"  Winter COP (Dec-Feb): {annual_stats['winter_cop']:.2f}")
    print(f"  Summer COP (Jun-Aug): {annual_stats['summer_cop']:.2f}")
    print(f"  Heating Season Energy: {annual_stats['heating_season_energy']:.1f} MWh")
    print(f"  Non-Heating Season Energy: {annual_stats['non_heating_season_energy']:.1f} MWh")
    
    # Save results if requested
    if save_path:
        df_monthly.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")
    
    return df_monthly, annual_stats


def plot_monthly_energy_analysis(
    df_monthly: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive monthly energy analysis plots.
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Monthly Energy-Based Heat Pump Analysis', fontsize=16, fontweight='bold')
    
    months = df_monthly['Month']
    month_names = [name[:3] for name in df_monthly['Month_Name']]
    
    # Plot 1: Energy consumption with HP/Pump breakdown
    ax1 = axes[0, 0]
    width = 0.6
    
    # Create stacked bar chart
    ax1.bar(months, df_monthly['Thermal_Energy_MWh'], width,
            label='Thermal Energy Output', color='red', alpha=0.7)
            
    ax1.bar(months, df_monthly['HP_Electrical_MWh'], width, 
            label='HP Electrical Energy', color='blue', alpha=0.8)
    ax1.bar(months, df_monthly['Pump_Electrical_MWh'], width, 
            label='Pump Electrical Energy', color='lightblue', alpha=0.8)

    
    ax1.set_title('Monthly Energy Consumption Breakdown')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Energy [MWh]')
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: COP comparison
    ax2 = axes[0, 1]
    ax2.plot(months, df_monthly['COP_Including_Pump'], 'o-', label='COP (incl. Pump)', linewidth=2, markersize=8)
    ax2.plot(months, df_monthly['COP_HP_Only'], 's-', label='COP (HP only)', linewidth=2, markersize=6)
    ax2.set_title('Monthly Energy-Based COP')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('COP')
    ax2.set_xticks(months)
    ax2.set_xticklabels(month_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add seasonal backgrounds
    ax2.axvspan(11.5, 2.5, alpha=0.1, color='blue', label='Winter')
    ax2.axvspan(5.5, 8.5, alpha=0.1, color='orange', label='Summer')
    
    # Plot 3: Activity rate
    ax3 = axes[1, 0]
    bars = ax3.bar(months, df_monthly['Activity_Rate'], color='green', alpha=0.7)
    ax3.set_title('Monthly Activity Rate')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Activity Rate [%]')
    ax3.set_xticks(months)
    ax3.set_xticklabels(month_names)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, df_monthly['Activity_Rate']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Annual COP Trend with benchmarks
    ax4 = axes[1, 1]
    
    # Energy efficiency ratio (COP)
    efficiency_ratio = df_monthly['Thermal_Energy_MWh'] / df_monthly['Total_Electrical_MWh']
    
    # Plot monthly COP trend
    ax4.plot(months, efficiency_ratio, 'o-', color='darkgreen', linewidth=3, markersize=10, 
             label='Monthly Energy COP', markerfacecolor='lightgreen', markeredgecolor='darkgreen', markeredgewidth=2)
    
    # Calculate and show annual average
    annual_avg_cop = df_monthly['Thermal_Energy_MWh'].sum() / df_monthly['Total_Electrical_MWh'].sum()
    ax4.axhline(y=annual_avg_cop, color='red', linestyle='--', linewidth=2, 
                label=f'Annual Average COP: {annual_avg_cop:.2f}')
    
    # Add seasonal averages
    winter_months = df_monthly[df_monthly['Month'].isin([12, 1, 2])]
    summer_months = df_monthly[df_monthly['Month'].isin([6, 7, 8])]
    
    winter_cop = (winter_months['Thermal_Energy_MWh'].sum() / 
                  winter_months['Total_Electrical_MWh'].sum())
    summer_cop = (summer_months['Thermal_Energy_MWh'].sum() / 
                  summer_months['Total_Electrical_MWh'].sum())
    
    # Add seasonal backgrounds
    ax4.axvspan(11.5, 2.5, alpha=0.1, color='blue')
    ax4.axvspan(5.5, 8.5, alpha=0.1, color='orange')
    
    ax4.set_title('Annual Energy COP Trend', fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Energy-Based COP')
    ax4.set_xticks(months)
    ax4.set_xticklabels(month_names)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')
    
    # Set y-axis limits to show variation better
    cop_min, cop_max = efficiency_ratio.min(), efficiency_ratio.max()
    y_margin = (cop_max - cop_min) * 0.15
    ax4.set_ylim(cop_min - y_margin, cop_max + y_margin)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Energy analysis plot saved to: {save_path}")
    
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List, Tuple, Union

class HeatPumpPlotter:
    """
    Modular heat pump analysis plotting class with extensive customization options.
    """
    
    def __init__(self, style: str = 'default', figsize_default: Tuple[int, int] = (10, 6)):
        """
        Initialize the plotter with default settings.
        
        Args:
            style: Matplotlib style to use
            figsize_default: Default figure size
        """
        plt.style.use(style)
        self.figsize_default = figsize_default
        self.colors = {
            'thermal': '#CC071E',
            'hp_electrical': '#00549F', 
            'pump_electrical': '#006165',
            'cop_total': '#000000',
            'cop_hp': '#CC071E',
            'activity': '#DDA0DD',
            'trend': '#000000',
            'winter': '#8EBAE5',
            'summer': '#FABE50'
        }
    
    def plot_energy_breakdown(
        self,
        df_monthly: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        colors: Optional[Dict[str, str]] = None,
        title: str = 'Monthly Energy Consumption Breakdown',
        xlabel: str = 'Month',
        ylabel: str = 'Energy [MWh]',
        bar_width: float = 0.6,
        alpha: float = 0.7,
        show_grid: bool = True,
        grid_alpha: float = 0.3,
        legend_loc: str = 'upper right',
        save_path: Optional[str] = None,
        dpi: int = 300,
        show_values: bool = False,
        value_format: str = '.1f'
    ) -> plt.Figure:
        """
        Create energy consumption breakdown plot.
        
        Args:
            df_monthly: DataFrame with monthly data
            figsize: Figure size tuple
            colors: Custom color dictionary
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            bar_width: Width of bars
            alpha: Transparency
            show_grid: Whether to show grid
            grid_alpha: Grid transparency
            legend_loc: Legend location
            save_path: Path to save figure
            dpi: DPI for saved figure
            show_values: Whether to show values on bars
            value_format: Format string for values
        """
        figsize = figsize or self.figsize_default
        colors = colors or self.colors
        
        fig, ax = plt.subplots(figsize=figsize)
        
        months = df_monthly['Month']
        month_names = [name[:3] for name in df_monthly['Month_Name']]
        
        # Create stacked bar chart
        ax.bar(months, df_monthly['Thermal_Energy_MWh'], bar_width,
               label='Thermal Energy Output', color=colors['thermal'], alpha=alpha)
               
        ax.bar(months, df_monthly['HP_Electrical_MWh'], bar_width, 
               label='HP Electrical Energy', color=colors['hp_electrical'], alpha=alpha)
        ax.bar(months, df_monthly['Pump_Electrical_MWh'], bar_width, 
               label='Pump Electrical Energy', color=colors['pump_electrical'], alpha=alpha)
        
        # Add values on bars if requested
        if show_values:
            for i, month in enumerate(months):
                thermal_val = df_monthly.iloc[i]['Thermal_Energy_MWh']
                ax.text(month, thermal_val + 0.5, f'{thermal_val:{value_format}}', 
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        ax.legend(loc=legend_loc)
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Energy breakdown plot saved to: {save_path}")
        
        return fig
    
    def plot_cop_comparison(
        self,
        df_monthly: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        colors: Optional[Dict[str, str]] = None,
        title: str = 'Monthly Energy-Based COP Comparison',
        xlabel: str = 'Month',
        ylabel: str = 'COP',
        line_width: float = 2.5,
        marker_size: int = 8,
        show_seasonal_bg: bool = True,
        seasonal_alpha: float = 0.1,
        show_grid: bool = True,
        grid_alpha: float = 0.3,
        legend_loc: str = 'upper right',
        save_path: Optional[str] = None,
        dpi: int = 300,
        y_margin_factor: float = 0.1
    ) -> plt.Figure:
        """
        Create COP comparison plot.
        """
        figsize = figsize or self.figsize_default
        colors = colors or self.colors
        
        fig, ax = plt.subplots(figsize=figsize)
        
        months = df_monthly['Month']
        month_names = [name[:3] for name in df_monthly['Month_Name']]
        
        # Plot COP lines
        ax.plot(months, df_monthly['COP_Including_Pump'], 'o-', 
               label='COP (incl. Pump)', linewidth=line_width, markersize=marker_size,
               color=colors['cop_total'])
        ax.plot(months, df_monthly['COP_HP_Only'], 's-', 
               label='COP (HP only)', linewidth=line_width, markersize=marker_size-2,
               color=colors['cop_hp'])
        
        # Add seasonal backgrounds
        if show_seasonal_bg:
            ax.axvspan(11.5, 12.5, alpha=seasonal_alpha, color='blue')
            ax.axvspan(0.5, 2.5, alpha=seasonal_alpha, color='blue')
            ax.axvspan(5.5, 8.5, alpha=seasonal_alpha, color='orange')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        ax.legend(loc=legend_loc)
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        
        # Set y-axis limits with margin
        cop_values = pd.concat([df_monthly['COP_Including_Pump'], df_monthly['COP_HP_Only']])
        y_min, y_max = cop_values.min(), cop_values.max()
        y_margin = (y_max - y_min) * y_margin_factor
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"COP comparison plot saved to: {save_path}")
        
        return fig
    
    def plot_activity_rate(
        self,
        df_monthly: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        color: str = None,
        title: str = 'Monthly Activity Rate',
        xlabel: str = 'Month',
        ylabel: str = 'Activity Rate [%]',
        bar_width: float = 0.6,
        alpha: float = 0.7,
        show_values: bool = True,
        value_format: str = '.1f',
        value_offset: float = 0.5,
        show_grid: bool = True,
        grid_alpha: float = 0.3,
        save_path: Optional[str] = None,
        dpi: int = 300
    ) -> plt.Figure:
        """
        Create activity rate plot.
        """
        figsize = figsize or self.figsize_default
        color = color or self.colors['activity']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        months = df_monthly['Month']
        month_names = [name[:3] for name in df_monthly['Month_Name']]
        
        bars = ax.bar(months, df_monthly['Activity_Rate'], width=bar_width,
                     color=color, alpha=alpha)
        
        # Add value labels on bars
        if show_values:
            for bar, value in zip(bars, df_monthly['Activity_Rate']):
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + value_offset, 
                       f'{value:{value_format}}%', 
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Activity rate plot saved to: {save_path}")
        
        return fig
    
    def plot_cop_trend_analysis(
        self,
        df_monthly: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = None,
        colors: Optional[Dict[str, str]] = None,
        title: str = 'Annual Energy COP Trend Analysis',
        xlabel: str = 'Month',
        ylabel: str = 'Energy-Based COP',
        line_width: float = 3,
        marker_size: int = 10,
        show_annual_avg: bool = True,
        show_seasonal_avg: bool = True,
        show_seasonal_bg: bool = True,
        seasonal_alpha: float = 0.1,
        show_grid: bool = True,
        grid_alpha: float = 0.3,
        legend_loc: str = 'upper right',
        save_path: Optional[str] = None,
        dpi: int = 300,
        y_margin_factor: float = 0.15
    ) -> plt.Figure:
        """
        Create comprehensive COP trend analysis plot.
        """
        figsize = figsize or self.figsize_default
        colors = colors or self.colors
        
        fig, ax = plt.subplots(figsize=figsize)
        
        months = df_monthly['Month']
        month_names = [name[:3] for name in df_monthly['Month_Name']]
        
        # Calculate efficiency ratio (COP)
        efficiency_ratio = df_monthly['Thermal_Energy_MWh'] / df_monthly['Total_Electrical_MWh']
        
        # Plot monthly COP trend
        ax.plot(months, efficiency_ratio, 'o-', color=colors['trend'], 
               linewidth=line_width, markersize=marker_size, 
               label='Monthly Energy COP', markerfacecolor='lightgreen', 
               markeredgecolor='darkgreen', markeredgewidth=2)
        
        # Calculate and show annual average
        if show_annual_avg:
            annual_avg_cop = (df_monthly['Thermal_Energy_MWh'].sum() / 
                            df_monthly['Total_Electrical_MWh'].sum())
            ax.axhline(y=annual_avg_cop, color='red', linestyle='--', linewidth=2, 
                      label=f'Annual Average COP: {annual_avg_cop:.2f}')
        
        # Add seasonal averages
        if show_seasonal_avg:
            winter_months = df_monthly[df_monthly['Month'].isin([12, 1, 2])]
            summer_months = df_monthly[df_monthly['Month'].isin([6, 7, 8])]
            
            if len(winter_months) > 0:
                winter_cop = (winter_months['Thermal_Energy_MWh'].sum() / 
                            winter_months['Total_Electrical_MWh'].sum())
                ax.axhline(y=winter_cop, color='blue', linestyle=':', linewidth=1.5,
                          label=f'Winter Avg: {winter_cop:.2f}')
            
            if len(summer_months) > 0:
                summer_cop = (summer_months['Thermal_Energy_MWh'].sum() / 
                            summer_months['Total_Electrical_MWh'].sum())
                ax.axhline(y=summer_cop, color='orange', linestyle=':', linewidth=1.5,
                          label=f'Summer Avg: {summer_cop:.2f}')
        
        # Add seasonal backgrounds
        if show_seasonal_bg:
            ax.axvspan(11.5, 12.5, alpha=seasonal_alpha, color='blue')
            ax.axvspan(0.5, 2.5, alpha=seasonal_alpha, color='blue')
            ax.axvspan(5.5, 8.5, alpha=seasonal_alpha, color='orange')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names)
        ax.legend(loc=legend_loc)
        
        if show_grid:
            ax.grid(True, alpha=grid_alpha)
        
        # Set y-axis limits to show variation better
        cop_min, cop_max = efficiency_ratio.min(), efficiency_ratio.max()
        y_margin = (cop_max - cop_min) * y_margin_factor
        ax.set_ylim(cop_min - y_margin, cop_max + y_margin)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"COP trend analysis plot saved to: {save_path}")
        
        return fig
    
    def create_all_plots(
        self,
        df_monthly: pd.DataFrame,
        save_directory: Optional[str] = None,
        file_prefix: str = 'hp_analysis',
        file_format: str = 'png',
        show_plots: bool = True,
        plot_specific_kwargs: Optional[Dict[str, Dict]] = None,
        **global_kwargs
    ) -> Dict[str, plt.Figure]:
        """
        Create all heat pump analysis plots.
        
        Args:
            df_monthly: Monthly data DataFrame
            save_directory: Directory to save plots
            file_prefix: Prefix for saved files
            file_format: File format for saved plots
            show_plots: Whether to display plots
            plot_specific_kwargs: Dict with plot-specific parameters
                e.g., {'energy_breakdown': {'legend_loc': 'lower left'}, 
                       'cop_comparison': {'legend_loc': 'upper right'}}
            **global_kwargs: Global arguments passed to all plot functions
        
        Returns:
            Dictionary of plot names and figure objects
        """
        figures = {}
        plot_specific_kwargs = plot_specific_kwargs or {}
        
        # Define plot configurations
        plot_configs = {
            'energy_breakdown': {
                'func': self.plot_energy_breakdown,
                'filename': f'{file_prefix}_energy_breakdown.{file_format}'
            },
            'cop_comparison': {
                'func': self.plot_cop_comparison,
                'filename': f'{file_prefix}_cop_comparison.{file_format}'
            },
            'activity_rate': {
                'func': self.plot_activity_rate,
                'filename': f'{file_prefix}_activity_rate.{file_format}'
            },
            'cop_trend': {
                'func': self.plot_cop_trend_analysis,
                'filename': f'{file_prefix}_cop_trend.{file_format}'
            }
        }
        
        for plot_name, config in plot_configs.items():
            save_path = None
            if save_directory:
                import os
                save_path = os.path.join(save_directory, config['filename'])
            
            # Combine global kwargs with plot-specific kwargs
            combined_kwargs = global_kwargs.copy()
            if plot_name in plot_specific_kwargs:
                combined_kwargs.update(plot_specific_kwargs[plot_name])
            
            # Create the plot
            fig = config['func'](df_monthly, save_path=save_path, **combined_kwargs)
            figures[plot_name] = fig
            
            if not show_plots:
                plt.close(fig)
        
        if show_plots:
            plt.show()
        
        return figures

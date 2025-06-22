import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Tuple


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

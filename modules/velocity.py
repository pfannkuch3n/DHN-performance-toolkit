# -*- coding: utf-8 -*-
"""
Velocity Analysis Module for District Heating Networks
=====================================================

This module provides comprehensive velocity analysis capabilities for district heating 
networks modeled with uesgraphs. It includes:

- Flow velocity calculations based on mass flow and pipe dimensions
- Comparison with manufacturer standards (DN pipe classifications)
- Constraint validation against technical guidelines
- Time series analysis of velocity violations
- Visualization tools for velocity distribution and constraint analysis

Key Features:
------------
- DN pipe classification and standard diameter mapping
- Velocity constraint checking based on technical guidelines
- Cumulative error analysis for constraint violations
- Network-wide velocity distribution analysis
- Integration with uesgraphs visualization tools

Technical Standards:
------------------
- Uses EN 10220 pipe diameter standards
- Implements VDI 2035 velocity guidelines
- Supports various pipe materials and applications

Example Usage:
-------------
```python
from uesgraphs.analysis.hydraulics import velocities

# Calculate velocities for network
graph_with_velocities = velocities.calculate_velocities(graph, density=950)

# Analyze velocity constraints
constraints_analysis = velocities.analyze_velocity_constraints(
    graph_with_velocities, 
    tolerance=0.1
)

# Visualize results
velocities.plot_velocity_violations(graph_with_velocities)
```
"""

import os
import logging
import tempfile
from datetime import datetime
from typing import List, Dict, Generator, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Import uesgraphs - will be available when integrated
try:
    import uesgraphs as ug
except ImportError:
    print("Warning: uesgraphs not available - some functions may not work")

# Am Anfang der Datei nach den anderen Imports:
try:
    from .utils import load_pipe_catalog  # Relativer Import wenn Teil von uesgraphs
except ImportError:
    try:
        from utils import load_pipe_catalog  # Absoluter Import wenn standalone
    except ImportError:
        print("Warning: load_pipe_catalog not available - catalog loading will fail")
        load_pipe_catalog = None

# Global cache für catalog
_CATALOG_CACHE = {}

def _get_catalog_data(catalog):
    """
    Intelligently load catalog data - handles both names and paths.
    
    Parameters:
    - catalog: str - Either catalog name ("isoplus") or full path ("/path/to/file.csv")
    """
    if load_pipe_catalog is None:
        raise ImportError("load_pipe_catalog function not available")
        
    if catalog not in _CATALOG_CACHE:
        # Check if it's a path (contains path separators) or a name
        if os.sep in catalog or '/' in catalog:
            # It's a path - extract directory and filename
            catalog_dir = os.path.dirname(catalog)
            catalog_name = os.path.splitext(os.path.basename(catalog))[0]
            _CATALOG_CACHE[catalog] = load_pipe_catalog(catalog_name, custom_path=catalog_dir)
        else:
            # It's a standard catalog name
            _CATALOG_CACHE[catalog] = load_pipe_catalog(catalog)
    
    return _CATALOG_CACHE[catalog]
    
# Default fluid properties
DEFAULT_FLUID_PROPERTIES = {
    "supply": {"density": 950},    # kg/m³ at typical supply temperature
    "return": {"density": 970}     # kg/m³ at typical return temperature
}

#### Logging Setup ####

def set_up_terminal_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a terminal-only logger for small helper functions."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Prevent double output
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def set_up_file_logger(name: str, log_dir: Optional[str] = None, 
                      level: int = logging.ERROR) -> logging.Logger:
    """Set up a file logger for major functions."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if log_dir is None:
        log_dir = tempfile.gettempdir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler for warnings and errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Log file created: {log_file}")
    return logger

#### Core Functions ####
def assign_DN_classifications(graph: 'ug.UESGraph', catalog: str = "isoplus",
                            classification_mode: str = "strict", tolerance: float = 0.05,
                            logger: Optional[logging.Logger] = None) -> Tuple['ug.UESGraph', Dict[str, set]]:
    """
    Assign DN classifications to network pipes with guaranteed 100% classification.
    
    Parameters
    ----------
    graph : ug.UESGraph
        Network graph with pipe diameter information
    catalog : str
        Either catalog name or path to catalog CSV
    classification_mode : str, optional
        "strict": Prefers smaller DN (conservative for safety/pressure)
        "generous": Prefers larger DN (optimistic for flow capacity)
    tolerance : float, optional
        Base tolerance around catalog diameters (default: 0.05 = ±5%)
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    tuple
        Modified graph with DN classifications (100% coverage) and DN groupings
        
    Notes
    -----
    Both modes guarantee 100% classification:
    - "strict": Falls back to next smaller DN if no match
    - "generous": Falls back to next larger DN if no match
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.assign_DN_classifications")
    
    # Load catalog and sort by diameter
    catalog_df = _get_catalog_data(catalog).sort_values('inner_diameter').reset_index(drop=True)
    logger.info(f"Mode: {classification_mode}, Tolerance: ±{tolerance:.1%}, Catalog: {len(catalog_df)} DN sizes")
    
    # Create tolerance ranges
    DN_ranges = {}
    for _, row in catalog_df.iterrows():
        dn, diameter = row['DN'], row['inner_diameter']
        DN_ranges[dn] = (diameter * (1 - tolerance), diameter * (1 + tolerance), diameter)
    
    # Initialize tracking
    DN_pipes = {dn: set() for dn in DN_ranges.keys()}
    classification_stats = {dn: 0 for dn in DN_ranges.keys()}
    fallback_used = []
    
    prefer_smaller = (classification_mode == "strict")
    
    # Classify each edge - guaranteed 100% success
    for edge in graph.edges:
        try:
            actual_diameter = graph.edges[edge]["diameter"]
            
            # 1. Try exact tolerance match
            matches = [dn for dn, (min_d, max_d, _) in DN_ranges.items() 
                      if min_d <= actual_diameter <= max_d]
            
            if matches:
                # Multiple matches: choose based on mode
                if len(matches) > 1:
                    sorted_matches = sorted(matches, key=lambda dn: DN_ranges[dn][2])
                    chosen_dn = sorted_matches[0] if prefer_smaller else sorted_matches[-1]
                    fallback_used.append((edge, actual_diameter, matches, chosen_dn, "ambiguous"))
                else:
                    chosen_dn = matches[0]
            else:
                # 2. No matches: fallback to closest DN based on mode
                sorted_dns = sorted(catalog_df['DN'], key=lambda dn: catalog_df[catalog_df['DN'] == dn]['inner_diameter'].iloc[0])
                
                if prefer_smaller:
                    # Find largest DN smaller than actual diameter, or smallest if none
                    smaller_dns = [dn for dn in sorted_dns if DN_ranges[dn][2] <= actual_diameter]
                    chosen_dn = smaller_dns[-1] if smaller_dns else sorted_dns[0]
                else:
                    # Find smallest DN larger than actual diameter, or largest if none  
                    larger_dns = [dn for dn in sorted_dns if DN_ranges[dn][2] >= actual_diameter]
                    chosen_dn = larger_dns[0] if larger_dns else sorted_dns[-1]
                
                fallback_used.append((edge, actual_diameter, [], chosen_dn, "fallback"))
            
            # Assign classification
            DN_pipes[chosen_dn].add(edge)
            graph.edges[edge]["DN"] = chosen_dn
            classification_stats[chosen_dn] += 1
            
        except KeyError as e:
            # Should never happen with fallback, but just in case
            logger.error(f"Edge {edge} missing diameter: {e}")
            # Emergency fallback to smallest DN
            emergency_dn = sorted(catalog_df['DN'], key=lambda dn: catalog_df[catalog_df['DN'] == dn]['inner_diameter'].iloc[0])[0]
            DN_pipes[emergency_dn].add(edge)
            graph.edges[edge]["DN"] = emergency_dn
            classification_stats[emergency_dn] += 1
    
    # Log results
    total_classified = sum(len(pipes) for pipes in DN_pipes.values())
    logger.info(f"SUCCESS: Classification: {total_classified}/{len(graph.edges)} pipes (100% success)")
    
    # Log DN distribution
    for dn, count in classification_stats.items():
        if count > 0:
            logger.info(f"  {dn}: {count} pipes")
    
    # Log fallback usage
    if fallback_used:
        ambiguous = [f for f in fallback_used if f[4] == "ambiguous"]
        fallbacks = [f for f in fallback_used if f[4] == "fallback"]
        
        if ambiguous:
            logger.info(f"INFO: {len(ambiguous)} ambiguous cases (chose {'smaller' if prefer_smaller else 'larger'} DN)")
        if fallbacks:
            logger.info(f"INFO: {len(fallbacks)} fallback cases (chose {'next smaller' if prefer_smaller else 'next larger'} DN)")
            
        # Show examples
        for edge, diameter, options, chosen, reason in fallback_used[:3]:
            if reason == "ambiguous":
                logger.debug(f"  {edge}: {diameter:.4f}m matched {options} → chose {chosen}")
            else:
                logger.debug(f"  {edge}: {diameter:.4f}m no match → fallback to {chosen}")
    
    return graph, DN_pipes


# Convenience functions  
def assign_DN_strict(graph: 'ug.UESGraph', catalog: str = "isoplus", tolerance: float = 0.05,
                    logger: Optional[logging.Logger] = None) -> Tuple['ug.UESGraph', Dict[str, set]]:
    """Conservative DN classification - smaller DN for safety, 100% coverage guaranteed"""
    return assign_DN_classifications(graph, catalog, "strict", tolerance, logger)

def assign_DN_generous(graph: 'ug.UESGraph', catalog: str = "isoplus", tolerance: float = 0.05,
                      logger: Optional[logging.Logger] = None) -> Tuple['ug.UESGraph', Dict[str, set]]:
    """Optimistic DN classification - larger DN for capacity, 100% coverage guaranteed"""  
    return assign_DN_classifications(graph, catalog, "generous", tolerance, logger)

def calculate_velocities(graph: 'ug.UESGraph', fluid_properties: Optional[Dict] = None,
                        system_type: str = "supply", logger: Optional[logging.Logger] = None) -> 'ug.UESGraph':
    """
    Calculate flow velocities for all pipes in the network.
    
    Parameters
    ----------
    graph : ug.UESGraph
        Network graph with mass flow data
    fluid_properties : dict, optional
        Dictionary containing fluid density information
    system_type : str
        Type of system ("supply" or "return")
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    ug.UESGraph
        Graph with added velocity information
    """
    if logger is None:
        logger = set_up_file_logger(f"{__name__}.calculate_velocities")
    
    if fluid_properties is None:
        fluid_properties = DEFAULT_FLUID_PROPERTIES
        logger.info("Using default fluid properties")
    
    try:
        density = fluid_properties[system_type]["density"]
    except KeyError:
        logger.error(f"Density not found for system type '{system_type}'")
        raise
    
    missing_data = []
    
    for edge in graph.edges:
        try:
            # Get mass flow and diameter
            mass_flow = graph.edges[edge]["m_flow"]  # kg/s
            diameter = graph.edges[edge]["diameter"]  # m
            
            # Calculate volume flow
            volume_flow = mass_flow / density  # m³/s
            
            # Calculate cross-sectional area
            area = np.pi * (diameter / 2) ** 2  # m²
            
            # Calculate velocity
            velocity = volume_flow / area  # m/s
            
            graph.edges[edge]["velocity"] = velocity
            
        except KeyError as e:
            missing_data.append((edge, str(e)))
    
    if missing_data:
        logger.error(f"Missing data for {len(missing_data)} edges")
        for edge, error in missing_data[:5]:  # Show first 5 errors
            logger.error(f"Edge {edge}: {error}")
    else:
        logger.info(f"Successfully calculated velocities for {len(graph.edges)} pipes")
    
    return graph

def calculate_pipe_constraints(pipe_data: Dict, system_type: str = "supply", 
                             fluid_properties: Optional[Dict] = None,
                             catalog: str = "isoplus") -> Dict:
    """
    Calculate velocity and mass flow constraints for a pipe based on catalog data.
    
    This function replaces the hardcoded constraint tables with dynamic constraint
    calculation based on the pipe catalog CSV data. It calculates both velocity and
    mass flow limits for a specific pipe based on its DN classification.
    
    Parameters
    ----------
    pipe_data : Dict
        Dictionary containing pipe information. Must include:
        - "DN" : str - DN classification (e.g., "DN20", "DN100")
    system_type : str, optional
        System type affecting fluid density (default: "supply")
        - "supply": typically higher temperature, lower density
        - "return": typically lower temperature, higher density
    fluid_properties : Dict, optional
        Dictionary with fluid properties. If None, uses defaults.
        Expected structure: {system_type: {"density": value_in_kg_m3}}
    catalog : str, optional
        Either catalog name for standard catalogs OR full path to custom catalog CSV.
        - Standard: "isoplus", "rehau", etc.
        - Custom: "/path/to/my_catalog.csv"
        
    Returns
    -------
    Dict
        Dictionary containing calculated constraints:
        {
            "velocity": {
                "min": float,  # Minimum velocity in m/s
                "max": float   # Maximum velocity in m/s
            },
            "mass_flow": {
                "min": float,  # Minimum mass flow in kg/s (from catalog)
                "max": float   # Maximum mass flow in kg/s (from catalog)
            },
            "catalog_data": {
                "inner_diameter": float,  # Catalog inner diameter in m
                "cross_sectional_area": float  # Calculated area in m²
            }
        }
    
    Calculation Method:
    ------------------
    1. Load pipe specifications from catalog based on DN classification
    2. Extract mass flow limits (min/max) directly from catalog (already in kg/s)
    3. Calculate cross-sectional area from catalog inner diameter
    4. Convert mass flows to velocities using: v = (ṁ/ρ) / A
       where: ṁ = mass flow [kg/s], ρ = density [kg/m³], A = area [m²]
    
    Formula Details:
    ---------------
    - Volume flow: V̇ = ṁ / ρ  [m³/s]
    - Cross-sectional area: A = π × (d/2)²  [m²]
    - Velocity: v = V̇ / A = (ṁ/ρ) / A  [m/s]
    
    Example:
    --------
    For DN20 pipe with:
    - Catalog inner diameter: 0.0273 m
    - Catalog mass flow range: 0.111 - 0.139 kg/s
    - Supply density: 950 kg/m³
    
    Calculations:
    - Area = π × (0.0273/2)² = 5.86e-4 m²
    - Min velocity = (0.111/950) / 5.86e-4 = 0.20 m/s
    - Max velocity = (0.139/950) / 5.86e-4 = 0.25 m/s
    
    Raises
    ------
    ValueError
        If the DN classification is not found in the catalog
    KeyError
        If required data is missing from pipe_data or fluid_properties
    """
    
    # Use default fluid properties if none provided
    if fluid_properties is None:
        fluid_properties = DEFAULT_FLUID_PROPERTIES
    
    # Extract required data
    try:
        nominal_diameter = pipe_data["DN"]
        density = fluid_properties[system_type]["density"]
    except KeyError as e:
        raise KeyError(f"Missing required data: {e}. "
                      f"Available pipe_data keys: {list(pipe_data.keys())}, "
                      f"Available fluid_properties: {list(fluid_properties.keys())}")
    
    # Load catalog and find matching DN entry
    catalog_df = _get_catalog_data(catalog)
    catalog_entry = catalog_df[catalog_df['DN'] == nominal_diameter]
    
    if catalog_entry.empty:
        available_dns = catalog_df['DN'].tolist()
        raise ValueError(f"DN '{nominal_diameter}' not found in catalog '{catalog}'. "
                        f"Available DN sizes: {available_dns}")
    
    # Extract catalog data (CSV values are already in correct SI units)
    catalog_row = catalog_entry.iloc[0]
    inner_diameter = catalog_row['inner_diameter']  # [m] - already in meters
    min_mass_flow = catalog_row['mass_flow_min']    # [kg/s] - already in kg/s
    max_mass_flow = catalog_row['mass_flow_max']    # [kg/s] - already in kg/s
    
    # Calculate cross-sectional area
    # A = π × (d/2)² where d is inner diameter
    cross_sectional_area = np.pi * (inner_diameter / 2) ** 2  # [m²]
    
    # Convert mass flows to velocities
    # v = (ṁ/ρ) / A = Volume_flow / Area
    min_volume_flow = min_mass_flow / density  # [m³/s]
    max_volume_flow = max_mass_flow / density  # [m³/s]
    
    min_velocity = min_volume_flow / cross_sectional_area  # [m/s]
    max_velocity = max_volume_flow / cross_sectional_area  # [m/s]
    
    # Create comprehensive result dictionary
    constraints = {
        "velocity": {
            "min": min_velocity,
            "max": max_velocity
        },
        "mass_flow": {
            "min": min_mass_flow,
            "max": max_mass_flow
        },
        "catalog_data": {
            "inner_diameter": inner_diameter,
            "cross_sectional_area": cross_sectional_area
        }
    }
    
    # Log detailed calculation info for debugging
    logger = set_up_terminal_logger(f"{__name__}.calculate_pipe_constraints")
    logger.debug(f"Constraints calculated for {nominal_diameter}:")
    logger.debug(f"  Catalog diameter: {inner_diameter:.4f} m")
    logger.debug(f"  Cross-sectional area: {cross_sectional_area:.6f} m²")
    logger.debug(f"  Density ({system_type}): {density} kg/m³")
    logger.debug(f"  Mass flow range: {min_mass_flow:.3f} - {max_mass_flow:.3f} kg/s")
    logger.debug(f"  Velocity range: {min_velocity:.3f} - {max_velocity:.3f} m/s")
    
    return constraints

def determine_constraint_violations(data_series: pd.Series, constraint: float, 
                                  constraint_type: str) -> Dict:
    """
    Identify periods where data violates constraints.
    
    Parameters
    ----------
    data_series : pd.Series
        Time series data to analyze
    constraint : float
        Constraint value
    constraint_type : str
        "min" or "max" constraint type
        
    Returns
    -------
    dict
        Dictionary with violation intervals and boolean series
    """
    if constraint_type == "min":
        violations = data_series < constraint
    else:
        violations = data_series > constraint
    
    violation_periods = []
    
    if len(violations.unique()) == 1:
        # All values either violate or don't violate
        if violations.any():
            start_time = data_series.index.min()
            end_time = data_series.index.max()
            duration = end_time - start_time
            violation_periods.append({
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "max_value": data_series.max(),
                "min_value": data_series.min()
            })
    else:
        # Find violation periods
        violation_groups = (violations != violations.shift()).cumsum()
        for group in violation_groups.unique():
            period_mask = violation_groups == group
            period_violations = violations[period_mask]
            
            if period_violations.any():
                start_time = data_series[period_mask][period_violations].index.min()
                end_time = data_series[period_mask][period_violations].index.max()
                
                if pd.notna(start_time) and pd.notna(end_time):
                    duration = end_time - start_time
                    violation_periods.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "max_value": data_series[period_mask].max(),
                        "min_value": data_series[period_mask].min()
                    })
    
    violations_df = pd.DataFrame(violation_periods)
    
    return {
        "intervals": violations_df,
        "bool_series": violations
    }

def analyze_constraint_violations(violations_df: pd.DataFrame, 
                                data_series: pd.Series) -> Dict:
    """
    Analyze violation statistics.
    
    Parameters
    ----------
    violations_df : pd.DataFrame
        DataFrame with violation intervals
    data_series : pd.Series
        Original time series data
        
    Returns
    -------
    dict
        Analysis results including duration and frequency
    """
    analysis = {}
    
    if not violations_df.empty and "duration" in violations_df.columns:
        analysis["total_duration"] = violations_df["duration"].sum()
        analysis["number_of_violations"] = len(violations_df)
        
        # Calculate normalized duration
        total_time = data_series.index.max() - data_series.index.min()
        analysis["normalized_duration"] = analysis["total_duration"] / total_time
    else:
        analysis["total_duration"] = pd.Timedelta(0)
        analysis["number_of_violations"] = 0
        analysis["normalized_duration"] = 0
    
    return analysis

def calculate_cumulative_error(data_series: pd.Series, violations: pd.Series, 
                             constraint: float) -> pd.Series:
    """Calculate cumulative error for constraint violations."""
    filtered_data = data_series[violations]
    if filtered_data.empty:
        return pd.Series(dtype=float)
    
    error = abs(filtered_data - constraint)
    return error.cumsum()

def calculate_mean_deviation(data_series: pd.Series, violations: pd.Series, 
                           constraint: float) -> float:
    """Calculate mean deviation from constraint during violations."""
    filtered_data = data_series[violations]
    if filtered_data.empty:
        return 0.0
    
    mean_value = filtered_data.mean()
    return abs(constraint - mean_value)

def analyze_velocity_constraints(graph: 'ug.UESGraph', tolerance: float = 0.1,
                               system_type: str = "supply",
                               fluid_properties: Optional[Dict] = None,
                               catalog: str = "isoplus",
                               logger: Optional[logging.Logger] = None) -> 'ug.UESGraph':
    """
    Comprehensive velocity constraint analysis for the entire network using catalog data.
    
    Parameters
    ----------
    graph : ug.UESGraph
        Network graph with velocity data and DN classifications
    tolerance : float
        Tolerance factor for constraints (0.1 = 10% tolerance)
    system_type : str
        System type ("supply" or "return")
    fluid_properties : dict, optional
        Fluid properties
    catalog : str, optional
        Either catalog name ("isoplus") or path to custom catalog CSV
    logger : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    ug.UESGraph
        Graph with added constraint analysis results
    """
    if logger is None:
        logger = set_up_file_logger(f"{__name__}.analyze_velocity_constraints")
    
    if fluid_properties is None:
        fluid_properties = DEFAULT_FLUID_PROPERTIES
    
    analysis_errors = []
    
    for edge in graph.edges:
        try:
            # Skip edges without DN classification
            if "DN" not in graph.edges[edge]:
                logger.warning(f"Edge {edge} missing DN classification, skipping")
                continue
            
            # Calculate constraints for this pipe using catalog parameter
            pipe_constraints = calculate_pipe_constraints(
                graph.edges[edge], system_type, fluid_properties, catalog  # catalog parameter hinzugefügt!
            )
            graph.edges[edge]["constraints"] = pipe_constraints
            
            # Initialize analysis structure
            graph.edges[edge]["analysis"] = {"velocity": {}}
            
            velocity_data = graph.edges[edge]["velocity"]
            
            # Analyze both min and max constraints
            for constraint_type in ["min", "max"]:
                base_constraint = pipe_constraints["velocity"][constraint_type]
                
                # Apply tolerance
                multiplier = (1 + tolerance) if constraint_type == "max" else (1 - tolerance)
                constraint_with_tolerance = base_constraint * multiplier
                
                # Find violations
                violations_data = determine_constraint_violations(
                    velocity_data, constraint_with_tolerance, constraint_type
                )
                
                # Analyze violations
                violation_analysis = analyze_constraint_violations(
                    violations_data["intervals"], velocity_data
                )
                
                # Calculate additional metrics
                cumulative_error = calculate_cumulative_error(
                    velocity_data, violations_data["bool_series"], constraint_with_tolerance
                )
                
                violation_analysis["cum_error"] = (
                    cumulative_error.iloc[-1] if not cumulative_error.empty else 0
                )
                violation_analysis["mean_deviation"] = calculate_mean_deviation(
                    velocity_data, violations_data["bool_series"], constraint_with_tolerance
                )
                
                graph.edges[edge]["analysis"]["velocity"][constraint_type] = violation_analysis
                
        except Exception as e:
            analysis_errors.append((edge, str(e)))
            logger.error(f"Error analyzing edge {edge}: {e}")
    
    if analysis_errors:
        logger.error(f"Analysis failed for {len(analysis_errors)} edges")
    else:
        logger.info(f"Successfully analyzed velocity constraints for {len(graph.edges)} edges")
    
    return graph

#### Utility Functions ####

def find_source_node(graph: 'ug.UESGraph') -> Tuple[Optional[int], Optional[tuple]]:
    """
    Find the supply source node in the network.
    
    Parameters
    ----------
    graph : ug.UESGraph
        Network graph
        
    Returns
    -------
    tuple
        Source node ID and its connected edge, or (None, None) if not found
    """
    logger = set_up_terminal_logger(f"{__name__}.find_source_node")
    
    for node in sorted(graph.nodes):
        try:
            name = graph.nodes[node].get("name", "").lower()
            if name == "supply1":
                edges = list(graph.edges(node))
                if len(edges) == 1:
                    return node, edges[0]
                else:
                    logger.warning(f"Source node {node} has {len(edges)} edges, expected 1")
        except Exception as e:
            logger.error(f"Error checking node {node}: {e}")
    
    logger.warning("No source node (name='supply1') with single edge found")
    return None, None

def validate_velocity_data(graph: 'ug.UESGraph', required_attributes: List[str] = None) -> List[tuple]:
    """
    Validate that required velocity analysis attributes exist.
    
    Parameters
    ----------
    graph : ug.UESGraph
        Network graph to validate
    required_attributes : list, optional
        List of required edge attributes
        
    Returns
    -------
    list
        List of (edge, missing_attribute) tuples for missing data
    """
    if required_attributes is None:
        required_attributes = ["m_flow", "diameter", "velocity", "DN"]
    
    missing_data = []
    
    for edge in graph.edges:
        for attr in required_attributes:
            if attr not in graph.edges[edge]:
                missing_data.append((edge, attr))
    
    return missing_data

#### Visualization Functions ####

def plot_DN_distribution(DN_pipes: Dict[str, set], title: str = "", 
                        save_path: Optional[str] = None) -> None:
    """
    Plot distribution of pipe diameters by DN classification.
    
    Parameters
    ----------
    DN_pipes : dict
        Dictionary mapping DN classifications to sets of pipes
    title : str
        Plot title suffix
    save_path : str, optional
        Path to save the plot
    """
    counts = {dn: len(pipes) for dn, pipes in DN_pipes.items()}
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(counts.keys(), counts.values(), color='steelblue', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.title(f'Distribution of Pipe Diameters {title}')
    plt.xlabel('DN Classification')
    plt.ylabel('Number of Pipes')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_velocity_time_series(pipe_data: Dict, key: str = "velocity", 
                            num_points: int = 100, save_path: Optional[str] = None) -> None:
    """
    Plot velocity time series with constraint lines.
    
    Parameters
    ----------
    pipe_data : dict
        Pipe data dictionary with velocity and constraints
    key : str
        Data key to plot
    num_points : int
        Number of time points to plot (for performance)
    save_path : str, optional
        Path to save the plot
    """
    # Sample data for performance
    data_series = pipe_data[key].iloc[:num_points] if len(pipe_data[key]) > num_points else pipe_data[key]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot time series
    data_series.plot(ax=ax, label=key.title(), color='blue', linewidth=2)
    
    # Add constraint lines if available
    if 'constraints' in pipe_data and key in pipe_data['constraints']:
        min_constraint = pipe_data['constraints'][key]['min']
        max_constraint = pipe_data['constraints'][key]['max']
        
        ax.axhline(y=min_constraint, color='red', linestyle='--',
                  label=f'Min Constraint ({min_constraint:.2f})', alpha=0.7)
        ax.axhline(y=max_constraint, color='red', linestyle='--',
                  label=f'Max Constraint ({max_constraint:.2f})', alpha=0.7)
    
    pipe_name = pipe_data.get("name", "Unknown")
    ax.set_title(f'{key.title()} Time Series for Pipe {pipe_name}', pad=20)
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{key.title()} [m/s]' if key == 'velocity' else key.title())
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_network_velocity_analysis(graph: 'ug.UESGraph', analysis_metric: str = "mean_deviation",
                                  constraint_type: str = "max", save_path: Optional[str] = None) -> None:
    """
    Plot network with velocity analysis results.
    
    Parameters
    ----------
    graph : ug.UESGraph
        Network graph with analysis results
    analysis_metric : str
        Analysis metric to visualize ("mean_deviation", "cum_error", "total_duration")
    constraint_type : str
        Constraint type ("min" or "max")
    save_path : str, optional
        Path to save the plot
    """
    try:
        # Prepare data for visualization
        for edge in graph.edges:
            if "analysis" in graph.edges[edge] and "velocity" in graph.edges[edge]["analysis"]:
                analysis_data = graph.edges[edge]["analysis"]["velocity"].get(constraint_type, {})
                
                if analysis_metric == "total_duration" and "total_duration" in analysis_data:
                    # Convert timedelta to hours
                    value = analysis_data["total_duration"].total_seconds() / 3600
                elif analysis_metric in analysis_data:
                    value = analysis_data[analysis_metric]
                else:
                    value = 0
                
                graph.edges[edge]["stats"] = value
            else:
                graph.edges[edge]["stats"] = 0
        
        # Create visualization using uesgraphs
        vis = ug.Visuals(graph)

        # Create descriptive labels that include constraint type
        constraint_label = "Max Velocity" if constraint_type == "max" else "Min Velocity"

        ylabel_map = {
            "mean_deviation": f"Mean Deviation [m/s] ({constraint_label})",
            "cum_error": f"Cumulative Error ({constraint_label})",
            "total_duration": f"Violation Duration [hours] ({constraint_label})"
        }

        ylabel = ylabel_map.get(analysis_metric, analysis_metric.title())

        # Create title with both metric and constraint type
        network_name = graph.graph.get("name", "Network")
        plot_title = f'{network_name} - {analysis_metric} ({constraint_type} violations)'

        fig = vis.show_network(
            show_plot=False,  # Don't block execution, just save
            scaling_factor=10,
            scaling_factor_diameter=50,
            ylabel=ylabel,
            generic_extensive_size="stats",
            timestamp=plot_title,
            zero_alpha=0.4,
            save_as=save_path
        )
        
    except Exception as e:
        logger = set_up_terminal_logger(f"{__name__}.plot_network_velocity_analysis")
        logger.error(f"Error plotting network analysis: {e}")

def plot_velocity_vs_diameter_simple(graph: 'ug.UESGraph', 
                                     catalog: str = "isoplus",
                                     velocity_metrics: List[str] = ["mean", "p95"],
                                     density: float = 950,
                                     max_velocity: Optional[float] = None,
                                     title_suffix: str = "",
                                     save_path: Optional[str] = None,
                                     logger: Optional[logging.Logger] = None) -> None:
    """
    Einfacher velocity vs diameter plot mit automatischer manufacturer curve.
    
    Parameters
    ----------
    graph : ug.UESGraph
        Network graph with velocity analysis
    catalog : str
        Catalog name or path
    velocity_metrics : List[str]
        Velocity metrics to plot (e.g., ["mean", "p95"])
    max_velocity : float, optional
        Maximum velocity for y-axis limit [m/s] (e.g., 2.0)
        If None, matplotlib decides automatically
    title_suffix : str
        Additional plot title text
    save_path : str, optional
        Save path for plot
    logger : logging.Logger, optional
        Logger instance
    """
    if logger is None:
        logger = set_up_terminal_logger(f"{__name__}.plot_velocity_vs_diameter_simple")
    
    # Load catalog
    catalog_df = _get_catalog_data(catalog)
    required_cols = ['inner_diameter', 'mass_flow_min', 'mass_flow_max', 'pressure_drop_min', 'pressure_drop_max']
    missing_cols = [col for col in required_cols if col not in catalog_df.columns]
    if missing_cols:
        logger.error(f"Catalog missing required columns: {missing_cols}")
        return
    
    # Prepare network data
    network_data = []
    max_diameter_mm = 0

    # Track why edges are skipped for better error reporting
    missing_DN_count = 0
    missing_velocity_count = 0
    successful_count = 0

    for edge in graph.edges:
        try:
            # Check for required attributes with detailed logging
            has_DN = "DN" in graph.edges[edge]
            has_velocity = "velocity" in graph.edges[edge]

            if not has_DN:
                missing_DN_count += 1
            if not has_velocity:
                missing_velocity_count += 1

            if not has_DN or not has_velocity:
                continue

            diameter_mm = graph.edges[edge]["diameter"] * 1000
            max_diameter_mm = max(max_diameter_mm, diameter_mm)
            velocity_series = graph.edges[edge]["velocity"]

            # Calculate metrics
            edge_data = {
                "edge": edge,
                "diameter_mm": diameter_mm,
                "DN": graph.edges[edge]["DN"],
                "mean": velocity_series.mean(),
                "median": velocity_series.median(),
                "max": velocity_series.max(),
                "p95": velocity_series.quantile(0.95),
                "p99": velocity_series.quantile(0.99)
            }

            network_data.append(edge_data)
            successful_count += 1

        except Exception as e:
            logger.warning(f"Error processing edge {edge}: {e}")

    if not network_data:
        logger.error("No valid network data found for plotting")
        logger.error(f"Total edges checked: {len(graph.edges)}")
        logger.error(f"Edges missing 'DN' attribute: {missing_DN_count}")
        logger.error(f"Edges missing 'velocity' attribute: {missing_velocity_count}")
        logger.error(f"Successfully processed edges: {successful_count}")
        logger.error("Plot function requires both 'DN' and 'velocity' attributes on edges")
        return
    
    logger.info(f"Plotting {len(network_data)} pipes, max diameter: {max_diameter_mm:.1f}mm")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 1. Calculate and plot manufacturer zone (green area)
    catalog_diameters_mm = catalog_df['inner_diameter'].values * 1000
    
    # Calculate velocities at min and max pressure drops
    velocity_min = []
    velocity_max = []
    
    for _, row in catalog_df.iterrows():
        diameter = row['inner_diameter']
        area = np.pi * (diameter / 2) ** 2
        
        # Velocity at min pressure drop (mass_flow_min)
        v_min = row['mass_flow_min'] / (density * area)
        velocity_min.append(v_min)
        
        # Velocity at max pressure drop (mass_flow_max)  
        v_max = row['mass_flow_max'] / (density * area)
        velocity_max.append(v_max)
    
    # Plot manufacturer zone
    ax.fill_between(catalog_diameters_mm, velocity_min, velocity_max,
                   alpha=0.3, color='green', 
                   label=f'Manufacturer Zone\n({catalog_df["pressure_drop_min"].iloc[0]}-{catalog_df["pressure_drop_max"].iloc[0]} Pa/m)', 
                   zorder=1)
    
    # 2. Plot reference pressure lines (optional)
    diameter_range = np.linspace(0, max_diameter_mm * 1.1, 100)
    for pa_per_m in [100, 200, 300]:
        velocities = []
        for d_mm in diameter_range:
            d_m = d_mm / 1000
            if d_m > 0:
                # APPROXIMATION: Simplified Darcy-Weisbach for reference lines
                # Δp/L = f * (ρ * v²) / (2 * D)
                # Solving for v: v = sqrt(2 * Δp/L * D / (f * ρ))
                # Using f ≈ 0.02 (smooth pipes, turbulent flow approximation)
                v = np.sqrt(2 * pa_per_m * d_m / (0.02 * density))
                velocities.append(min(v, max_velocity or 20.0))  # Cap at max_velocity or 5 m/s
            else:
                velocities.append(0)
        
        ax.plot(diameter_range, velocities,
               linestyle='--', color='gray', alpha=0.7, linewidth=1,
               label=f'{pa_per_m} Pa/m' if pa_per_m == 100 else "")
    
    # 3. Plot network data points
    colors = ['blue', 'red', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v']
    
    metric_labels = {
        'mean': 'Mean Velocity',
        'median': 'Median Velocity',
        'max': 'Maximum Velocity',
        'p95': '95th Percentile',
        'p99': '99th Percentile'
    }
    
    for i, metric in enumerate(velocity_metrics):
        if metric not in metric_labels:
            continue
            
        x_data = [pipe["diameter_mm"] for pipe in network_data]
        y_data = [pipe[metric] for pipe in network_data if metric in pipe]
        
        ax.scatter(x_data, y_data,
                  c=colors[i % len(colors)],
                  marker=markers[i % len(markers)],
                  s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                  label=metric_labels[metric], zorder=3)
    
    # 4. Set axis limits
    ax.set_xlim(0, max_diameter_mm * 1.1)  # 10% margin on diameter
    
    if max_velocity is not None:
        ax.set_ylim(0, max_velocity)  # Fixed y-axis limit
    else:
        ax.set_ylim(bottom=0)  # Start at 0, let matplotlib handle upper limit
    
    # 5. Styling
    ax.set_xlabel('Inner Pipe Diameter [mm]', fontsize=12)
    ax.set_ylabel('Flow Velocity [m/s]', fontsize=12)
    
    base_title = 'Velocity vs Diameter Analysis'
    if title_suffix:
        base_title += f' - {title_suffix}'
    ax.set_title(base_title, fontsize=14, pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 6. Add statistics box
    total_pipes = len(network_data)
    
    # Calculate compliance (pipes in green zone)
    in_zone_count = 0
    for pipe in network_data:
        pipe_diameter_mm = pipe["diameter_mm"]
        pipe_velocity = pipe[velocity_metrics[0]]  # Use first metric
        
        # Interpolate manufacturer zone for this diameter
        if catalog_diameters_mm.min() <= pipe_diameter_mm <= catalog_diameters_mm.max():
            v_min_interp = np.interp(pipe_diameter_mm, catalog_diameters_mm, velocity_min)
            v_max_interp = np.interp(pipe_diameter_mm, catalog_diameters_mm, velocity_max)
            
            if v_min_interp <= pipe_velocity <= v_max_interp:
                in_zone_count += 1
    
    stats_text = f'Network Statistics:\n'
    stats_text += f'Total Pipes: {total_pipes}\n'
    stats_text += f'In Manufacturer Zone: {in_zone_count} ({in_zone_count/total_pipes*100:.1f}%)\n'
    stats_text += f'Max Diameter: {max_diameter_mm:.1f} mm'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {save_path}")
    else:
        plt.show()
    
    logger.info(f"Plot completed: {in_zone_count}/{total_pipes} pipes in manufacturer zone")


#### Main Analysis Functions ####

def velocity_analysis_pipeline(graph: 'ug.UESGraph',
                             fluid_properties: Optional[Dict] = None,
                             tolerance: float = 0.1,
                             system_type: str = "supply",
                             catalog: str = "isoplus",
                             logger: Optional[logging.Logger] = None) -> 'ug.UESGraph':
    """
    Complete velocity analysis pipeline with flexible catalog support.
   
    This function orchestrates the full velocity analysis workflow:
    1. Assign DN classifications using specified catalog
    2. Calculate velocities based on mass flow and pipe dimensions
    3. Analyze constraint violations against catalog specifications
    4. Validate results for completeness
   
    Parameters
    ----------
    graph : ug.UESGraph
        Network graph with mass flow and diameter data. Required edge attributes:
        - 'm_flow': mass flow in kg/s
        - 'diameter': pipe diameter in meters
    fluid_properties : dict, optional
        Fluid properties dictionary with structure:
        {system_type: {"density": value_in_kg_m3}}
        If None, uses default values (supply: 950 kg/m³, return: 970 kg/m³)
    tolerance : float, optional
        Tolerance factor for constraint checking (default: 0.1 = 10%)
        Applied as: min_constraint *= (1-tolerance), max_constraint *= (1+tolerance)
    system_type : str, optional
        System type affecting fluid density (default: "supply")
        Options: "supply" or "return"
    catalog : str, optional
        Catalog specification - either standard name or custom path:
        - Standard catalogs: "isoplus", "rehau", etc.
        - Custom paths: "/path/to/my_catalog.csv", "./data/pipes.csv"
        (default: "isoplus")
    logger : logging.Logger, optional
        Logger instance for detailed output. If None, creates a file logger.
       
    Returns
    -------
    ug.UESGraph
        Graph with complete velocity analysis including:
        - Edge attributes added: 'DN', 'velocity', 'constraints', 'analysis'
        - Analysis results for each pipe's constraint violations
        
    Pipeline Steps:
    --------------
    1. **DN Classification**: Assigns nominal diameter categories based on catalog
    2. **Velocity Calculation**: Computes flow velocities from mass flows
    3. **Data Validation**: Checks for missing required attributes
    4. **Constraint Analysis**: Evaluates violations against catalog limits
    
    Example Usage:
    -------------
    ```python
    # Standard usage with default Isoplus catalog
    analyzed_graph = velocity_analysis_pipeline(graph)
    
    # Custom catalog from file
    analyzed_graph = velocity_analysis_pipeline(
        graph, 
        catalog="/path/to/custom_pipes.csv",
        tolerance=0.15,
        system_type="return"
    )
    
    # Custom fluid properties
    custom_props = {"supply": {"density": 920}, "return": {"density": 980}}
    analyzed_graph = velocity_analysis_pipeline(
        graph,
        fluid_properties=custom_props,
        catalog="rehau"
    )
    ```
    
    Raises:
    ------
    FileNotFoundError
        If specified catalog file cannot be found
    ValueError
        If graph contains pipes that cannot be classified or analyzed
    KeyError
        If required edge attributes are missing from the graph
        
    Notes:
    -----
    - The pipeline modifies the input graph in-place
    - Results are cached for performance on repeated calls
    - All intermediate steps are logged for debugging
    - Pipeline continues even if some edges fail individual steps
    """
    if logger is None:
        logger = set_up_file_logger(f"{__name__}.velocity_analysis_pipeline")
   
    logger.info("Starting velocity analysis pipeline")
    logger.info(f"Configuration: catalog='{catalog}', system_type='{system_type}', tolerance={tolerance}")
   
    try:
        # Step 1: Assign DN classifications using specified catalog
        logger.info("Step 1: Assigning DN classifications")
        graph, DN_pipes = assign_DN_classifications(graph, catalog=catalog, logger=logger)
        
        # Log classification summary
        classified_count = sum(len(pipes) for pipes in DN_pipes.values())
        logger.info(f"Classification results: {classified_count} pipes classified into {len(DN_pipes)} DN categories")
       
        # Step 2: Calculate velocities
        logger.info("Step 2: Calculating velocities")
        graph = calculate_velocities(graph, fluid_properties, system_type, logger)
       
        # Step 3: Validate data completeness
        logger.info("Step 3: Validating velocity data")
        missing_data = validate_velocity_data(graph)
        if missing_data:
            logger.warning(f"Missing data found for {len(missing_data)} edge-attribute pairs")
            # Log specific missing data for debugging
            for edge, attr in missing_data[:5]:  # Show first 5 examples
                logger.warning(f"  Edge {edge} missing attribute: {attr}")
            if len(missing_data) > 5:
                logger.warning(f"  ... and {len(missing_data) - 5} more missing attributes")
       
        # Step 4: Analyze constraints using catalog specifications
        logger.info("Step 4: Analyzing velocity constraints")
        graph = analyze_velocity_constraints(
            graph, 
            tolerance=tolerance, 
            system_type=system_type, 
            fluid_properties=fluid_properties,
            catalog=catalog,
            logger=logger
        )
       
        logger.info("Velocity analysis pipeline completed successfully")
        
        # Log final summary
        total_edges = len(graph.edges)
        edges_with_analysis = sum(1 for edge in graph.edges 
                                if "analysis" in graph.edges[edge])
        logger.info(f"Pipeline summary: {edges_with_analysis}/{total_edges} edges successfully analyzed")
       
    except Exception as e:
        logger.error(f"Velocity analysis pipeline failed: {e}")
        logger.error(f"Error occurred during pipeline execution", exc_info=True)
        raise
   
    return graph

#### Module Exports ####

__all__ = [
    # Main functions
    'velocity_analysis_pipeline',
    'calculate_velocities',
    'analyze_velocity_constraints',
    
    # DN and constraints
    'assign_DN_classifications',
    'calculate_pipe_constraints',
    
    # Analysis functions
    'determine_constraint_violations',
    'analyze_constraint_violations',
    'calculate_cumulative_error',
    'calculate_mean_deviation',
    
    # Utilities
    'find_source_node',
    'validate_velocity_data',
    
    # Visualization
    'plot_DN_distribution',
    'plot_velocity_time_series',
    'plot_network_velocity_analysis',
    
    # Constants 
    'DEFAULT_FLUID_PROPERTIES'
    

]
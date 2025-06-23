import pandas as pd
import re
from typing import Callable, Optional
from functools import reduce
import logging
import tempfile
import os

def load_pipe_catalog(catalog_name: str = "isoplus",custom_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load pipe catalog data from CSV file in the data/pipe_catalogs directory.
    
    This function loads manufacturer pipe catalog data containing pipe dimensions
    and flow capacities for different nominal diameters (DN). The catalog files
    are expected to be located in the data/pipe_catalogs subdirectory relative
    to the systemmodels module.
    
    Parameters
    ---------- 
    catalog_name : str, optional
        Name of the pipe catalog to load (default: "isoplus")
        The function will look for a file named "{catalog_name}.csv"
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing pipe catalog data with columns:
        - DN: Nominal diameter [designated, e.g. DN20]
        - wall_thickness: Pipe wall thickness [m] 
        - inner_diameter: Inner pipe diameter [m]
        - mass_flow_min: Minimum mass flow capacity [kg/s]
        - mass_flow_max: Maximum mass flow capacity [kg/s]
        
    Raises
    ------
    FileNotFoundError
        If the specified catalog file does not exist
    ValueError
        If the catalog file exists but contains invalid data structure
        
    Examples
    --------
    >>> catalog = load_pipe_catalog("isoplus")
    >>> print(catalog.columns.tolist())
    ['DN', 'wall_thickness', 'inner_diameter', 'mass_flow_min', 'mass_flow_max']
    
    >>> # Load different catalog (if available)
    >>> rehau_catalog = load_pipe_catalog("rehau")
    
    Notes
    -----
    The CSV files can contain comment lines starting with '#' which will be
    automatically ignored during loading. This allows for metadata and source
    information to be stored directly in the catalog files.
    
    The function expects the catalog files to be located at:
    {module_directory}/uesgraphs/data/pipe_catalogs/{catalog_name}.csv
    """
    
    if custom_path:
        catalog_path = os.path.join(custom_path, f"{catalog_name}.csv")
    else:
        # Einfacher relativer Pfad
        current_dir = os.path.dirname(os.path.abspath(__file__))
        catalog_path = os.path.join(current_dir, "..", "data", "pipe_catalogs", f"{catalog_name}.csv")
    
    # Convert to absolute path for better error reporting
    catalog_path = os.path.abspath(catalog_path)    

    if not os.path.exists(catalog_path):
        raise FileNotFoundError(f"Catalog '{catalog_name}' not found at: {catalog_path}")
    
    try:
        # Load CSV data, ignoring comment lines starting with '#'
        catalog_df = pd.read_csv(catalog_path, comment='#')
        
        # Validate required columns exist
        required_columns = ['DN', 'inner_diameter', 'mass_flow_min', 'mass_flow_max']
        missing_columns = [col for col in required_columns if col not in catalog_df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Catalog '{catalog_name}' is missing required columns: {missing_columns}\n"
                f"Available columns: {catalog_df.columns.tolist()}"
            )
        
        # Sort by DN for consistent ordering
        catalog_df = catalog_df.sort_values('inner_diameter').reset_index(drop=True)
        
        return catalog_df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"Catalog file '{catalog_name}' is empty or contains no valid data")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing catalog file '{catalog_name}': {str(e)}")



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
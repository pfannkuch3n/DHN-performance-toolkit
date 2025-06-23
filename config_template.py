"""
Configuration Template - Copy to config_local.py and adjust paths
"""

SCENARIOS = {
    "example_scenario": {
        "name": "Example DHN Analysis",
        "json_path": r"path\to\your\network.json",
        "data_path": r"path\to\your\simulation\data",
        "output_dir": r"outputs\example_scenario"
    },
    
    # Add your scenarios here:
    # "your_scenario": {
    #     "name": "Your Project Name",
    #     "json_path": r"C:\path\to\network.json",
    #     "data_path": r"C:\path\to\simulation_results",
    #     "output_dir": r"outputs\your_scenario"
    # }
}

# Analysis Parameters
DEFAULT_PARAMS = {
    "start_date": "2024-01-01",
    "end_date": "2024-12-31", 
    "time_interval": "15min",
    "time_interval_hours": 0.25,
    "pipe_catalog": "data/isoplus.csv"  # Uses the included catalog
}
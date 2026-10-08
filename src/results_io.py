"""Saving solver runs to disk and reading them back: stdout log files and JSON result files
(``<log_dir>/<city>_<solver>.json``) that hold the samples and QUBO parameters of each run.
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import numpy as np


@contextmanager
def create_log_context(log_file_path):
    """
    Context manager to redirect stdout to a log file.
    
    Parameters
    ----------
    log_file_path : str
        Path to the log file (will overwrite if exists).
    
    Yields
    ------
    file
        The opened log file handle.
    
    Example
    -------
    >>> with create_log_context('logs/Apulia_SimulatedAnnealing.log'):
    >>>     print("This goes to the log file")
    """
    log_file = open(log_file_path, 'w')
    old_stdout = sys.stdout
    sys.stdout = log_file
    try:
        yield log_file
    finally:
        sys.stdout = old_stdout
        log_file.close()


def save_results_json(results_dict, city, solver_name, log_dir, sample_data=None, qubo_parameters=None):
    """
    Save solver results as a JSON file.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing solver results with keys like:
        - solver: str, solver name
        - city: str, city name
        - timestamp: str, ISO format timestamp
        - timing: dict, timing information
        - results: dict, solution results
        - parameters: dict, solver parameters
    city : str
        Name of the city/network.
    solver_name : str
        Name of the solver.
    log_dir : str
        Directory to save the JSON file.
    sample_data : list of dict, optional
        List of sample dictionaries from extract_sample_data() containing:
        - 'solution': dict with variable assignments
        - 'energy': float, energy value
        - 'num_occurrences': int, occurrence count
    qubo_parameters : dict, optional
        QUBO problem parameters containing:
        - 'rho': float, penalty parameter
        - 's': int, target number of sensors
        - 'num_variables': int, number of variables in problem
    
    Returns
    -------
    str
        Path to the saved JSON file.
    """
    json_path = os.path.join(log_dir, f"{city}_{solver_name}.json")
    
    # Ensure results_dict has required structure
    if 'timestamp' not in results_dict:
        results_dict['timestamp'] = datetime.now().isoformat()
    if 'city' not in results_dict:
        results_dict['city'] = city
    if 'solver' not in results_dict:
        results_dict['solver'] = solver_name
    
    # Add sample data if provided
    if sample_data is not None:
        results_dict['samples'] = sample_data
        results_dict['sample_metadata'] = {
            'total_unique_samples': len(sample_data),
            'total_occurrences': sum(s['num_occurrences'] for s in sample_data)
        }
    
    # Add QUBO parameters if provided
    if qubo_parameters is not None:
        results_dict['qubo_parameters'] = qubo_parameters
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    results_dict = convert_to_serializable(results_dict)
    
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    return json_path


def extract_sample_data(sampleset):
    """
    Extract all unique sample solutions from an aggregated sampleset.
    
    Parameters
    ----------
    sampleset : dimod.SampleSet
        Aggregated sampleset from D-Wave solvers (e.g., simAnnSamples.aggregate()).
    
    Returns
    -------
    list of dict
        List of sample dictionaries, each containing:
        - 'solution': dict with variable assignments {0: 1, 1: 0, ...}
        - 'energy': float, energy value for this solution
        - 'num_occurrences': int, number of times this solution appeared
    
    Example
    -------
    >>> aggregated = simAnnSamples.aggregate()
    >>> samples = extract_sample_data(aggregated)
    >>> print(f"Extracted {len(samples)} unique solutions")
    """
    sample_data = []
    
    for sample, energy, num_occ in sampleset.data(['sample', 'energy', 'num_occurrences']):
        sample_dict = {
            'solution': {str(k): int(v) for k, v in sample.items()},  # Convert to JSON-serializable format
            'energy': float(energy),
            'num_occurrences': int(num_occ)
        }
        sample_data.append(sample_dict)
    
    return sample_data


def load_samples_from_json(json_path):
    """
    Load sample data from a saved JSON file.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON file containing solver results with sample data.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'samples': list of dicts with 'solution', 'energy', 'num_occurrences'
        - 'qubo_parameters': dict with 'rho', 's', 'num_variables'
        - 'solver': str, solver name
        - Other result metadata
    
    Example
    -------
    >>> data = load_samples_from_json('logs/Fossolo_SimulatedAnnealing.json')
    >>> print(f"Loaded {len(data['samples'])} unique samples")
    >>> print(f"QUBO params: rho={data['qubo_parameters']['rho']}, s={data['qubo_parameters']['s']}")
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'samples' not in data:
        raise ValueError(f"JSON file {json_path} does not contain 'samples' data. "
                         "Make sure the file was saved with the enhanced save_results_json().")
    
    if 'qubo_parameters' not in data:
        raise ValueError(f"JSON file {json_path} does not contain 'qubo_parameters'. "
                         "Make sure the file was saved with the enhanced save_results_json().")
    
    return data

"""Turning sampler output into tables and metrics: energy/probability tables, energy binning,
time to solution (TTS), the feasibility boundary and the approximation gap.
"""

import numpy as np
import pandas as pd


def samples_to_df(samples):
    """
    Convert samples list (from JSON) to a DataFrame suitable for plotting.
    
    Parameters
    ----------
    samples : list of dict
        List of sample dictionaries, each containing:
        - 'energy': float
        - 'num_occurrences': int
    
    Returns
    -------
    pd.DataFrame
        DataFrame with energy as index and probability as column.
    
    Example
    -------
    >>> data = load_samples_from_json('logs/Fossolo_SimulatedAnnealing.json')
    >>> df = samples_to_df(data['samples'])
    >>> print(df.head())
    """
    # Aggregate energies and their occurrences
    energy_counts = {}
    total_occurrences = 0
    
    for sample in samples:
        energy = sample['energy']
        num_occ = sample['num_occurrences']
        
        if energy in energy_counts:
            energy_counts[energy] += num_occ
        else:
            energy_counts[energy] = num_occ
        
        total_occurrences += num_occ
    
    # Convert to probabilities
    energy_probs = {energy: count / total_occurrences for energy, count in energy_counts.items()}
    
    # Create DataFrame
    df = pd.DataFrame.from_dict(energy_probs, orient='index', columns=['Probability']).sort_index()
    df.index.name = 'Energy'
    return df


def sampleset_to_df(results, skip=1, imported=False):
    """
    Convert the results of the optimization to a dataframe.
    Parameters
    ----------
    results : dimod.SampleSet or similar
        The results of the optimization.
    skip : int, optional
        parameter to avoid putting all xlabels, by default 1
    imported : bool, optional
        If True, expects results as a dict-like with 'energy' and 'num_occurrences'.
    Returns
    -------
    pd.DataFrame
        DataFrame with energy as index and probability as column.
    """
    if imported:
        energies = results['energy']
        occurrences = results['num_occurrences']
    else:
        energies = results.data_vectors['energy']
        occurrences = results.data_vectors['num_occurrences']
    counts = {}
    total = sum(occurrences)
    for index, energy in enumerate(energies):
        if energy in counts:
            counts[energy] += occurrences[index]
        else:
            counts[energy] = occurrences[index]
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.columns = ['Probability']
    df.index.name = 'Energy'
    return df


def sum_infeas_soln(df, threshold):
    """
    Consolidate the infeasible solutions in the results dataframe.
    Parameters
    ----------
    df : pd.DataFrame
        The results of the optimization.
    threshold : float
        The energy threshold for the optimal or feasible solution.
    Returns
    -------
    pd.DataFrame
        DataFrame with infeasible solutions summed and appended as a new row.
    """
    infeasible = df[df.index > threshold]
    infeasible_sum = infeasible.sum()
    infeasible_sum.name = threshold
    df = df[df.index <= threshold]
    df = pd.concat([df, infeasible_sum.to_frame().T])
    return df


def bin_energy_levels(df, bin_size=0.1, round_decimals=2):
    """
    Bin the energy levels of a dataframe to reduce clutter.
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe with energy levels as index and probabilities as values.
    bin_size : float
        The size of each bin.
    round_decimals : int
        The number of decimal places to round the energy levels to.
    Returns
    -------
    pd.DataFrame
        A dataframe with binned energy levels.
    """
    binned_index = (df.index.to_series() // bin_size * bin_size).round(round_decimals)
    binned_df = df.copy()
    binned_df.index = binned_index
    binned_df = binned_df.groupby(binned_df.index).sum()
    return binned_df


def calculate_tts(sample_set, s, energy_threshold, exec_time):
    """
    Calculate the time to solution (optimality or feasibility) for the simulated annealing solver.
    Parameters
    ----------
    sample_set : dimod.SampleSet or iterable
        The samples returned by the simulated or quantum annealing solver.
    s : int
        The expected total size for feasible solutions.
    energy_threshold : float
        The energy threshold for the optimal or feasible solution.
    exec_time : float
        The execution time of the simulated or quantum annealing solver.
    Returns
    -------
    tuple
        (TTS_opt, TTS_fea, p_opt, p_fea)
    """
    if hasattr(sample_set, 'data'):
        samples_data = list(sample_set.data(['sample', 'energy', 'num_occurrences']))
    else:
        samples_data = list(sample_set)
    tts = exec_time
    total_size = s
    p_opt = 0
    p_fea = 0
    n = 0
    for sample, energy, num_ocu in samples_data:
        if isinstance(sample, dict):
            total_size = sum(sample.values())
        elif hasattr(sample, 'values'):
            total_size = sum(sample.values())
        if total_size == s:
            p_fea += num_ocu
        if energy <= energy_threshold:
            p_opt += num_ocu
        n += num_ocu
    p_opt = p_opt/n if n else 0
    p_fea = p_fea/n if n else 0
    if p_opt == 1:
        TTS_opt = tts
    elif p_opt == 0:
        TTS_opt = np.inf
    else:
        TTS_opt = tts * np.log(1 - 0.99) / np.log(1 - p_opt)
    if p_fea == 1:
        TTS_fea = tts
    elif p_fea == 0:
        TTS_fea = np.inf
    else:
        TTS_fea = tts * np.log(1 - 0.99) / np.log(1 - p_fea)
    print(f'Probability of optimal solution <= {energy_threshold} : {p_opt}')
    print(f'Probability of feasible solution satisfying constraints : {p_fea}')
    print(f'Time to solution for optimal solution : {TTS_opt}')
    print(f'Time to solution for feasible solution : {TTS_fea}')
    return TTS_opt, TTS_fea, p_opt, p_fea


def calculate_feasibility_boundary(rho, s, df_list=None, method='theoretical'):
    """
    Calculate the feasibility boundary that separates feasible from infeasible solutions.
    
    In QUBO formulation:
    - Energy = objective_value + penalty_term
    - Penalty term = rho*(sum(x_i) - s)^2
    - For feasible solutions (sum(x_i) = s): penalty = 0
    - For 1 sensor deviation: penalty = rho
    
    Therefore, feasibility boundary = min_feasible_energy + rho
    
    Solutions with energy < boundary are FEASIBLE (satisfy constraint)
    Solutions with energy >= boundary are INFEASIBLE (violate constraint)
    
    Parameters
    ----------
    rho : float
        The penalty parameter from QUBO formulation.
    s : int
        The target number of sensors.
    df_list : list of pd.DataFrame, optional
        List of dataframes with energy levels as index for empirical method.
    method : str, optional
        Method to calculate boundary: 'theoretical', 'empirical', or 'both'.
        Default is 'theoretical'.
    
    Returns
    -------
    float or dict
        The feasibility boundary energy value. If method='both', returns dict with
        {'theoretical': value, 'empirical': value, 'boundaries_per_solver': list}.
    
    Notes
    -----
    - Theoretical: boundary = min_energy + rho (minimum penalty for 1 sensor deviation)
    - Empirical: finds largest energy gap in the data distribution (may not be accurate)
    """
    if method == 'theoretical' or method == 'both':
        # Theoretical boundary: minimum energy where penalty starts (deviation by 1)
        if df_list is not None and len(df_list) > 0:
            min_energy = min([df.index.min() for df in df_list if len(df) > 0])
            theoretical_boundary = min_energy + rho
        else:
            # Fallback: return just rho as relative boundary
            theoretical_boundary = rho
    
    if method == 'empirical' or method == 'both':
        if df_list is None or len(df_list) == 0:
            raise ValueError("df_list must be provided for empirical method")
        
        empirical_boundaries = []
        for df in df_list:
            if len(df) > 1:
                energies = sorted(df.index)
                gaps = [energies[i+1] - energies[i] for i in range(len(energies)-1)]
                max_gap_idx = gaps.index(max(gaps))
                # Boundary is in the middle of the largest gap
                boundary = (energies[max_gap_idx] + energies[max_gap_idx + 1]) / 2
                empirical_boundaries.append(boundary)
        
        empirical_boundary = np.mean(empirical_boundaries) if empirical_boundaries else None
    
    if method == 'theoretical':
        return theoretical_boundary
    elif method == 'empirical':
        return empirical_boundary
    elif method == 'both':
        result = {
            'theoretical': theoretical_boundary,
            'empirical': empirical_boundary,
            'boundaries_per_solver': empirical_boundaries if 'empirical_boundaries' in locals() else []
        }
        return result
    else:
        raise ValueError("method must be 'theoretical', 'empirical', or 'both'")


def calculate_approximation_ratio(glob_Obj, best_Obj):
    """
    Calculates the approximation ratio between the global objective value and the best known objective value.
    Args:
        glob_Obj (float): The global objective value obtained from the optimization.
        best_Obj (float): The best found objective value with a method for comparison.
    Returns:
        float: The approximation ratio, calculated as glob_Obj / best_Obj.
    Raises:
        ValueError: If best_Obj is zero to avoid division by zero.
    Examples:
        >>> ratio = calculate_approximation_ratio(150, 100)
        >>> print(ratio)
        1.5
    """
    if best_Obj == 0:
        raise ValueError("best_Obj cannot be zero to avoid division by zero.")
    return 1 - (glob_Obj / best_Obj)

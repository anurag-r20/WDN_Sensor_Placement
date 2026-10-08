"""Figures for the notebook: the network and sensor placements, sampler energy distributions, the
feasibility boundary across solvers, the coverage curve and the TTS comparison.
"""

import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .analysis import bin_energy_levels, calculate_feasibility_boundary, samples_to_df, sum_infeas_soln
from .results_io import load_samples_from_json


def save_current_figure(fig_path, dpi=300, bbox_inches='tight', close_fig=False):
    """
    Save the current matplotlib figure to a file.
    
    Parameters
    ----------
    fig_path : str
        Path where the figure should be saved.
    dpi : int, optional
        Resolution in dots per inch (default: 300).
    bbox_inches : str, optional
        Bounding box specification (default: 'tight').
    close_fig : bool, optional
        Whether to close the figure after saving (default: False).
    
    Returns
    -------
    str
        Path to the saved figure.
    """
    # Get current figure - create one if none exists
    fig = plt.gcf()
    if fig.get_axes():  # Check if figure has any axes (i.e., has content)
        fig.savefig(fig_path, dpi=dpi, bbox_inches=bbox_inches)
        if close_fig:
            plt.close(fig)
    else:
        print(f"Warning: No active figure to save at {fig_path}")
    return fig_path


def plot_WDN(G, VC, EB, city, show=True):
    """Plots the graph of the city's Water Distribution Network (WDN).
    This function visualizes the WDN graph using NetworkX and Matplotlib. It plots nodes with their
    positions, edges, and labels for both nodes and edges. It also includes a legend and title for the plot.
    Args:
        G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
        VC (dict): Vertex cost dictionary for node labels.
        EB (dict): Edge betweenness dictionary for edge labels.
        city (str): Name of the city (for plot title/logging).
        show (bool): Whether to display the plot (default: True).
    Returns:
        None
    Raises:
        ValueError: If the graph `G` is None or the position dictionary is empty.
    Examples:
        >>> plot_WDN(G, VC, EB, city)
    """
    
    if G is None:
        print(f"Graph could not be constructed for {city}.")
        return
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        print("Error: Position dictionary is empty.")
        return
    # Create plot
    plt.figure(figsize=(24, 16))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
    nx.draw_networkx_edges(G, pos, edge_color='black')
    edge_labels_rounded = {k: round(v, 2) for k, v in EB.items()}
    node_labels_rounded = {k: round(v, 2) for k, v in VC.items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_rounded, label_pos=0.5, font_size=25, verticalalignment='bottom')
    # nx.draw_networkx_labels(G, pos, labels=node_labels_rounded, font_size=25, font_color='black')
    blue_patch = Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=30, label='nodes = source / sink / junction')
    black_line = Line2D([0], [0], color='black', lw=5, label='pipes')
    plt.legend(handles=[blue_patch, black_line], loc='lower left', fontsize='35')
    # plt.title(f"Water Distribution Network - {city}", fontsize = 80)
    if show:
        plt.show()


def plot_sensor_placement(G, VC, EB, city, sensor_placement, title):
    """Plots a city's Water Distribution Network (WDN) and the optimal sensor placement.
    This function visualizes the WDN graph with nodes colored based on sensor placement: red for nodes with
    sensors and sky blue for nodes without. It uses Matplotlib to create the plot and adds a legend to distinguish
    between sensor and non-sensor nodes.
    Args:
        G (nx.Graph): The NetworkX graph object for the WDN.
        VC (dict): Vertex cost dictionary for node labels.
        EB (dict): Edge betweenness dictionary for edge labels.
        city (str): The name of the city for which the WDN is plotted.
        sensor_placement (dict): A dictionary indicating sensor placement at each node (1 for sensor, 0 for no sensor).
        title (str): The title of the plot.
    Returns:
        None
    Raises:
        ValueError: If the graph `G` is None.
    Examples:
        >>> plot_sensor_placement(G, VC, EB, 'Apulian', {'J1': 1, 'J2': 0, 'J3': 1}, 'Optimal Sensor Placement')
    """

    if G is None:
        print(f"Graph could not be constructed for {city}.")
        return
    pos = nx.get_node_attributes(G, 'pos')
    node_colors = ['red' if sensor_placement.get(node, 0) == 1 else 'skyblue' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edge_color='black')
    nx.draw_networkx_labels(G, pos, font_size=6, font_color='black')
    edge_labels_rounded = {k: round(v, 2) for k, v in EB.items()}
    node_labels_rounded = {k: round(v, 2) for k, v in VC.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels_rounded, label_pos=0.5, font_size=6, verticalalignment='bottom')
    nx.draw_networkx_labels(G, pos, labels=node_labels_rounded, font_size=8, font_color='black')
    red_patch = Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=30, label='Sensor placed')
    blue_patch = Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=30, label='No sensor')
    plt.legend(handles=[red_patch, blue_patch], loc='best', fontsize='120')
    # plt.title(title, fontsize=80)
    plt.show()


def plot_comparison(G, VC, EB, city, sensor_placement_MIQP, show=True):
    """Plots a city's Water Distribution Network (WDN) with optimal sensor placements from both MIP and QUBO formulations.
    This function generates a single figure with two subplots, showing the WDN with sensor placements obtained
    through the MIP and QUBO models. It utilizes the `plot_sensor_placement` function to visualize the results.
    Args:
        G (nx.Graph): The NetworkX graph object for the WDN.
        VC (dict): Vertex cost dictionary for node labels.
        EB (dict): Edge betweenness dictionary for edge labels.
        city (str): The name of the city for which the WDN is plotted.
        sensor_placement_MIQP (dict): A dictionary indicating sensor placement at each node from the MIP formulation.
        show (bool): Whether to display the plot (default: True).
    Returns:
        None
    Examples:
        >>> plot_comparison(G, VC, EB, 'Apulian', {'J1': 1, 'J2': 0, 'J3': 1})
    """
    plt.figure(figsize=(80, 60))
    plot_sensor_placement(G, VC, EB, city, sensor_placement_MIQP, f'Optimal Sensor Placement - {city}')
    if show:
        plt.show()


def plot_energies(results, title=None, skip=1, show=True):
    
    if hasattr(results, 'data_vectors'):
        energies = results.data_vectors['energy']
        occurrences = results.data_vectors['num_occurrences']
    else:
        energies = [datum.energy for datum in results.data(['energy'], sorted_by='energy')]
        occurrences = [1] * len(energies)
    counts = Counter()
    total = sum(occurrences)
    for index, energy in enumerate(energies):
        counts[energy] += occurrences[index]
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    ax = df.plot(kind='bar', legend=None)
    plt.xlabel('Energy')
    plt.ylabel('Probabilities')
    ax.set_xticklabels([t if not i%skip else "" for i,t in enumerate(ax.get_xticklabels())])
    plt.title(str(title))
    if show:
        plt.show()
    print("minimum energy:", min(energies))


def plot_samples(results, title=None):

    if hasattr(results, 'vartype') and str(results.vartype) == 'Vartype.BINARY':
        samples = [''.join(c for c in str(datum.sample.values()).strip(', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        plt.xlabel('bitstring for solution')
    else:
        samples = list(range(len(results)))
        plt.xlabel('solution')
    counts = Counter(samples)
    total = len(samples)
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None)
    plt.xticks(rotation=80)
    plt.ylabel('Probabilities')
    plt.title(str(title))
    plt.show()
    
    # If energy info is available, print min energy
    if hasattr(results, 'data_vectors') and 'energy' in results.data_vectors:
        energies = results.data_vectors['energy']
        print("minimum energy:", min(energies))


def plot_enumerate(results, title=None, show=True):
    
    energies = [datum.energy for datum in results.data(['energy'], sorted_by='energy')]
    if hasattr(results, 'vartype') and str(results.vartype) == 'Vartype.BINARY':
        samples = [''.join(c for c in str(datum.sample.values()).strip(', ') if c.isdigit()) for datum in results.data(['sample'], sorted_by=None)]
        plt.xlabel('bitstring for solution')
    else:
        samples = list(range(len(energies)))
        plt.xlabel('solution')
    plt.bar(samples, energies)
    plt.xticks(rotation=90)
    plt.ylabel('Energy')
    plt.title(str(title))
    print("minimum energy:", min(energies))
    if show:
        plt.show()


def plot_multibar_graph_discrete(df_list, df_names, df_colors, vertical_line=None, skip=1, title=None, round_decimals=2, bin_size=0.1, include_gurobi=True, feasibility_boundary=None, show=True):
    """
    Plot the multibar graph of the results of the optimization with discrete (evenly spaced) x-axis,
    ensuring unique energy levels with rounding and binning, and leaving gaps for large differences in integer parts.
    Parameters
    ----------
    df_list : list of pd.DataFrame
        The list of dataframes to plot.
    df_names : list of str
        The names of the dataframes.
    df_colors : list of str
        The bar colors of the dataframes.
    vertical_line : float, optional
        The energy level to plot a vertical line at (Gurobi solution), by default None.
    skip : int, optional
        The number of x-ticks to skip, by default 1.
    title : str, optional
        The plot title, by default None.
    round_decimals : int, optional
        The number of decimal places to round the energy levels to, by default 2.
    bin_size : float, optional
        The size of each bin to aggregate energy levels, by default 0.1.
    include_gurobi : bool, optional
        Whether to include Gurobi reference lines, by default True.
    feasibility_boundary : float, optional
        The energy level for the feasibility boundary line, by default None.
    """

    # Bin energy levels in all DataFrames
    binned_dfs = [bin_energy_levels(df, bin_size, round_decimals) for df in df_list]
    # Combine all energy levels (treat as discrete categories)
    energy_levels = sorted(set.union(*(set(df.index) for df in binned_dfs)))
    # Adjust for large gaps between integer parts of energy levels
    adjusted_energy_levels = []
    prev_level = None
    gap_threshold = 1  # Threshold for gaps between integer parts
    for level in energy_levels:
        if prev_level is not None:
            if int(level) - int(prev_level) >= gap_threshold:
                adjusted_energy_levels.append(prev_level + (level - prev_level) / 2)
        adjusted_energy_levels.append(level)
        prev_level = level
    unique_dfs = []
    for df in binned_dfs:
        aligned_df = pd.DataFrame({'Energy': adjusted_energy_levels}).set_index('Energy').join(df, how='left').fillna(0)
        unique_dfs.append(aligned_df)
    
    width = 0.8 / len(df_list)
    pos = np.arange(len(adjusted_energy_levels))
    fig, ax = plt.subplots(figsize=(18, 8))
    
    for i, (df, name, color) in enumerate(zip(unique_dfs, df_names, df_colors)):
        ax.bar(pos + i * width, df['Probability'], width, alpha=0.7, color=color, label=name)
    
    # Plot Gurobi reference line if enabled
    if include_gurobi and vertical_line is not None:
        binned_vertical_line = round(vertical_line // bin_size * bin_size, round_decimals)
        if binned_vertical_line in adjusted_energy_levels:
            gurobi_idx = adjusted_energy_levels.index(binned_vertical_line)
            # Center the line on the bar group
            gurobi_x = gurobi_idx + (len(df_list) - 1) * width / 2
            ax.axvline(x=gurobi_x, color='black', linestyle='--', label='Gurobi')
    
    # Plot feasibility boundary if provided
    if feasibility_boundary is not None:
        # Find the closest energy level to the boundary
        binned_boundary = round(feasibility_boundary // bin_size * bin_size, round_decimals)
        
        # Check if boundary is within data range
        min_energy = adjusted_energy_levels[0]
        max_energy = adjusted_energy_levels[-1]
        
        if feasibility_boundary > max_energy:
            # Boundary is to the right of all data - ALL SAMPLES ARE FEASIBLE
            print(f"[INFO] Feasibility boundary ({feasibility_boundary:.4f}) is above max energy ({max_energy:.4f})")
            print(f"[INFO] All observed solutions are FEASIBLE (satisfy constraint)")
            
            # Draw boundary line at the right edge (centered on last bar group)
            boundary_x = len(adjusted_energy_levels) - 1 + (len(df_list) - 1) * width / 2 + 0.5
            ax.axvline(x=boundary_x, color='orange', linestyle='--', linewidth=2.5, 
                       label=f'Feasibility Boundary ({feasibility_boundary:.2f})', zorder=10)
            
            # Shade entire region as feasible (green)
            ax.axvspan(-0.5, boundary_x, alpha=0.1, color='green', zorder=0)
            
            # Add annotation
            ax.text(len(adjusted_energy_levels) * 0.5, ax.get_ylim()[1] * 0.5, 
                    'All Solutions\nFEASIBLE', 
                    fontsize=20, ha='center', va='center', color='darkgreen', 
                    weight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        elif feasibility_boundary < min_energy:
            # Boundary is to the left of all data - ALL SAMPLES ARE INFEASIBLE
            print(f"[INFO] Feasibility boundary ({feasibility_boundary:.4f}) is below min energy ({min_energy:.4f})")
            print(f"[INFO] All observed solutions are INFEASIBLE (violate constraint)")
            
            # Draw boundary line at the left edge
            boundary_x = -0.5
            ax.axvline(x=boundary_x, color='orange', linestyle='--', linewidth=2.5, 
                       label=f'Feasibility Boundary ({feasibility_boundary:.2f})', zorder=10)
            
            # Shade entire region as infeasible (red)
            last_x = len(adjusted_energy_levels) - 1 + (len(df_list) - 1) * width / 2 + 0.5
            ax.axvspan(boundary_x, last_x, alpha=0.1, color='red', zorder=0)
            
            # Add annotation
            ax.text(len(adjusted_energy_levels) * 0.5, ax.get_ylim()[1] * 0.5, 
                    'All Solutions\nINFEASIBLE', 
                    fontsize=20, ha='center', va='center', color='darkred', 
                    weight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        else:
            # Boundary is within data range - MIXED FEASIBLE/INFEASIBLE
            if binned_boundary in adjusted_energy_levels:
                boundary_idx = adjusted_energy_levels.index(binned_boundary)
            else:
                # Find the nearest energy level
                closest_level = min(adjusted_energy_levels, key=lambda x: abs(x - binned_boundary))
                boundary_idx = adjusted_energy_levels.index(closest_level)
            
            print(f"[INFO] Feasibility boundary ({feasibility_boundary:.4f}) within data range [{min_energy:.4f}, {max_energy:.4f}]")
            print(f"[INFO] Solutions with energy < {feasibility_boundary:.4f} are FEASIBLE")
            print(f"[INFO] Solutions with energy >= {feasibility_boundary:.4f} are INFEASIBLE")
            
            # Draw vertical line for boundary (centered on the bar group)
            boundary_x = boundary_idx + (len(df_list) - 1) * width / 2
            ax.axvline(x=boundary_x, color='orange', linestyle='--', linewidth=2.5, 
                       label=f'Feasibility Boundary ({feasibility_boundary:.2f})', zorder=10)
            
            # Add shaded regions
            last_x = len(adjusted_energy_levels) - 1 + (len(df_list) - 1) * width / 2 + 0.5
            ax.axvspan(-0.5, boundary_x, alpha=0.1, color='green', zorder=0)
            ax.axvspan(boundary_x, last_x, alpha=0.1, color='red', zorder=0)
            
            # Add text annotations
            if boundary_idx > len(adjusted_energy_levels) * 0.3:
                ax.text(boundary_idx * 0.5, ax.get_ylim()[1] * 0.5, 'Feasible\nRegion', 
                        fontsize=18, ha='center', va='center', color='darkgreen', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if boundary_idx < len(adjusted_energy_levels) * 0.7:
                ax.text((boundary_idx + len(adjusted_energy_levels)) * 0.5, ax.get_ylim()[1] * 0.5, 
                        'Infeasible\nRegion', fontsize=18, ha='center', va='center', 
                        color='darkred', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_yscale('log')
    ax.set_ylabel('Probability', fontsize=25)
    ax.set_xlabel('Objective Value', fontsize=25)
    ax.set_ylim(1e-5, 1e0)
    xtick_labels = [f"{e:.2f}" if i % skip == 0 else "" for i, e in enumerate(adjusted_energy_levels)]
    # Center the ticks on the bar groups (offset by half the total bar width)
    tick_positions = pos + (len(df_list) - 1) * width / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=25)
    ax.tick_params(axis='y', labelsize=25)
    plt.legend(loc='upper center', ncol=len(df_list), fontsize=15)
    plt.title(title, fontsize=25)
    if show:
        plt.show()


def plot_coverage(s, objective_value_list_MIP):
    """
    Plots the objective value against the number of sensors and marks the minimum objective value.
    Args:
        s (list): The list of sensor counts.
        objective_value_list_MIP (list): A list of objective values obtained from the MIP formulation.
    Returns:
        None
    Example:
        >>> plot_coverage([0, 1, 2, 3, 4, 5], [100, 95, 90, 85, 80, 75])
    """
    min_obj_value = min(objective_value_list_MIP)
    min_index = objective_value_list_MIP.index(min_obj_value)
    plt.figure(figsize=(16, 16))
    plt.plot(s, objective_value_list_MIP, marker='o', linestyle='-', color='b', label='Objective Value')
    plt.plot(s[min_index], min_obj_value, marker='x', markersize=10, color='r', label=f'Min Objective Value: {min_obj_value}')
    plt.title('Objective Value vs Number of Sensors')
    plt.xlabel('Number of Sensors (s)')
    plt.ylabel('Objective Value')
    plt.xticks(s)  # Ensures that the x-axis shows each integer sensor count
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_tts_comparison(nodes, gurobi_MIP, sim_ann_best, tabu_search_best,
                        gurobi_QUBO, sim_ann_feasible, tabu_search_feasible, include_gurobi=True):
    """
    Plot Time-to-Solution (TTS) comparison across different solvers and problem sizes.
    
    This function creates a log-scale plot comparing TTS for both best (optimal) and feasible 
    solutions across different optimization methods (Gurobi MIP, Gurobi QUBO, Simulated Annealing, 
    and Tabu Search) as a function of network node size.
    
    Args:
        nodes (list): List of node sizes for different WDN instances.
        gurobi_MIP (list): TTS values for Gurobi MIP solver (best solutions).
        sim_ann_best (list): TTS values for Simulated Annealing (best solutions).
        tabu_search_best (list): TTS values for Tabu Search (best solutions).
        gurobi_QUBO (list): TTS values for Gurobi QUBO solver (best solutions).
        sim_ann_feasible (list): TTS values for Simulated Annealing (feasible solutions).
        tabu_search_feasible (list): TTS values for Tabu Search (feasible solutions).
        include_gurobi (bool, optional): Whether to include Gurobi results in plot, by default True.
    
    Returns:
        None
    
    Example:
        >>> nodes = [24, 37, 114, 272, 856]
        >>> gurobi_MIP = [0.06, 0.06, 0.08, 0.03, 0.11]
        >>> sim_ann_best = [np.inf, np.inf, np.inf, np.inf, np.inf]
        >>> tabu_search_best = [84.3, 84.3, 10700, 5750, 254000]
        >>> gurobi_QUBO = [238, 43700, 43700, 43700, 43700]
        >>> sim_ann_feasible = [2.61, 2.24, 23.9, 119, 1300]
        >>> tabu_search_feasible = [84.3, 84.3, 84.2, 84.6, 110]
        >>> plot_tts_comparison(nodes, gurobi_MIP, sim_ann_best, tabu_search_best,
        ...                     gurobi_QUBO, sim_ann_feasible, tabu_search_feasible, include_gurobi=False)
    """
    plt.figure(figsize=(16, 12))
    
    # Increase line width for better visibility
    line_width = 3
    marker_size = 13
    
    # Plot for Best Objective Solutions (Optimal)
    if include_gurobi:
        plt.plot(nodes, gurobi_QUBO, label='Gurobi (QUBO_Best)', marker='o', linestyle='-', 
                 color='brown', linewidth=line_width, markersize=marker_size)
    plt.plot(nodes, tabu_search_best, label='Tabu Search (Best)', marker='o', linestyle='-', 
             color='blue', linewidth=line_width, markersize=marker_size)
    if include_gurobi:
        plt.plot(nodes, gurobi_MIP, label='Gurobi (MIP_Best)', marker='o', linestyle='-', 
                 color='black', linewidth=line_width, markersize=marker_size)
    plt.plot(nodes, sim_ann_best, label='Simulated Annealing (Best)', marker='o', linestyle='-', 
             color='red', linewidth=line_width, markersize=marker_size)
    
    # Plot for Feasible Solutions
    plt.plot(nodes, tabu_search_feasible, label='Tabu Search (Feasible)', marker='x', linestyle='--', 
             color='blue', linewidth=line_width, markersize=marker_size)
    plt.plot(nodes, sim_ann_feasible, label='Simulated Annealing (Feasible)', marker='x', linestyle='--', 
             color='red', linewidth=line_width, markersize=marker_size)
    
    plt.yscale('log')  # Log scale for better visibility of large TTS differences
    plt.xlabel('Node Size', fontsize=30)
    plt.ylabel('TTS (s)', fontsize=30)
    
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    
    # Adjust the legend to have three columns and place it below the plot
    plt.legend(fontsize=18, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def load_and_plot_feasibility_from_json(log_dir, city, solver_names=None, threshold=100, 
                                         include_gurobi=False, gurobi_optimal=None,
                                         colors=None, save_path=None, show=True):
    """
    Load solver results from JSON files and create the feasibility boundary plot.
    
    Parameters
    ----------
    log_dir : str
        Directory containing the JSON log files.
    city : str
        City name used in the JSON filenames (e.g., 'Fossolo').
    solver_names : list of str, optional
        List of solver names to include. Default: ['LeapHybrid', 'QuantumAnnealing', 'SimulatedAnnealing', 'TabuSearch']
    threshold : float, optional
        Energy threshold for consolidating high-energy solutions. Default: 100
    include_gurobi : bool, optional
        Whether to include Gurobi optimal solution line. Default: False
    gurobi_optimal : float, optional
        Gurobi optimal energy value (if include_gurobi=True).
    colors : list of str, optional
        Colors for each solver. Default: ['green', 'blue', 'red', 'yellow']
    save_path : str, optional
        Path to save the plot. If None, uses log_dir/city_FeasibilityBoundary_FromJSON.png
    show : bool, optional
        Whether to display the plot. Default: True
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'dataframes': list of DataFrames for each solver
        - 'qubo_parameters': QUBO parameters from the first solver
        - 'feasibility_boundary': calculated feasibility boundary
        - 'solver_names': list of solver names used
    
    Example
    -------
    >>> result = load_and_plot_feasibility_from_json('logs', 'Fossolo')
    >>> print(f"Feasibility boundary: {result['feasibility_boundary']:.4f}")
    """
    if solver_names is None:
        solver_names = ['LeapHybrid', 'QuantumAnnealing', 'SimulatedAnnealing', 'TabuSearch']
    
    if colors is None:
        colors = ['green', 'blue', 'red', 'yellow']
    
    # Load data from JSON files
    df_list = []
    filtered_df_list = []
    qubo_params = None
    display_names = []
    
    for solver_name in solver_names:
        json_path = os.path.join(log_dir, f"{city}_{solver_name}.json")
        
        if not os.path.exists(json_path):
            print(f"⚠️  Warning: {json_path} not found, skipping {solver_name}")
            continue
        
        try:
            data = load_samples_from_json(json_path)
            df = samples_to_df(data['samples'])
            filtered_df = sum_infeas_soln(df, threshold)
            
            df_list.append(df)
            filtered_df_list.append(filtered_df)
            
            # Store QUBO parameters from first successful load
            if qubo_params is None:
                qubo_params = data['qubo_parameters']
            
            # Create display name (convert CamelCase to spaces)
            display_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', solver_name)
            display_names.append(display_name)
            
            print(f"✓ Loaded {solver_name}: {len(data['samples'])} unique samples")
            
        except Exception as e:
            print(f"❌ Error loading {solver_name}: {e}")
            continue
    
    if not filtered_df_list:
        raise ValueError("No valid JSON files found. Cannot create plot.")
    
    if qubo_params is None:
        raise ValueError("Could not extract QUBO parameters from any JSON file.")
    
    # Calculate feasibility boundary
    rho = qubo_params['rho']
    s = qubo_params['s']
    
    feasibility_boundary_data = calculate_feasibility_boundary(rho, s, filtered_df_list, method='both')
    min_energy_data = min([df.index.min() for df in filtered_df_list if len(df) > 0])
    max_energy_data = max([df.index.max() for df in filtered_df_list if len(df) > 0])
    
    print(f"\n=== Feasibility Boundary Analysis (from JSON) ===")
    print(f"Rho (penalty parameter): {rho}")
    print(f"Target sensors (s): {s}")
    print(f"Min energy in data: {min_energy_data:.4f}")
    print(f"Max energy in data: {max_energy_data:.4f}")
    print(f"Theoretical feasibility boundary: {feasibility_boundary_data['theoretical']:.4f}")
    print(f"Empirical feasibility boundary: {feasibility_boundary_data['empirical']:.4f}")
    
    theoretical_boundary = feasibility_boundary_data['theoretical']
    if theoretical_boundary > max_energy_data:
        print(f"\n⚠️  IMPORTANT: Theoretical boundary ({theoretical_boundary:.4f}) is ABOVE max data ({max_energy_data:.4f})")
        print(f"    This means ALL samples are FEASIBLE (satisfy the constraint sum(x_i) = {s})")
        feasibility_boundary = theoretical_boundary
    elif theoretical_boundary < min_energy_data:
        print(f"\n⚠️  WARNING: Theoretical boundary ({theoretical_boundary:.4f}) is BELOW min data ({min_energy_data:.4f})")
        feasibility_boundary = theoretical_boundary
    else:
        print(f"\n✓ Theoretical boundary ({theoretical_boundary:.4f}) is within data range")
        feasibility_boundary = theoretical_boundary
    
    print(f"====================================\n")
    
    # Trim colors list to match number of solvers loaded
    colors_trimmed = colors[:len(filtered_df_list)]
    
    # Create plot
    plot_multibar_graph_discrete(
        filtered_df_list, display_names, colors_trimmed,
        vertical_line=gurobi_optimal if include_gurobi else None,
        skip=3,
        title=f'Solutions From Different Solvers',
        include_gurobi=include_gurobi,
        feasibility_boundary=feasibility_boundary,
        show=False
    )
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(log_dir, f"{city}_FeasibilityBoundary_FromJSON.png")
    
    save_current_figure(save_path)
    print(f"✓ Feasibility boundary plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        'dataframes': filtered_df_list,
        'qubo_parameters': qubo_params,
        'feasibility_boundary': feasibility_boundary,
        'solver_names': display_names,
        'feasibility_boundary_data': feasibility_boundary_data
    }

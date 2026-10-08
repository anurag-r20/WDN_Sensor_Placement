# Import necessary libraries
from networkx import edges
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import operator
import networkx as nx
from collections import Counter
import dimod
import neal
from tabu import TabuSampler
import pyomo.environ as pyo
from matplotlib.lines import Line2D
import json
import sys
from contextlib import contextmanager
from datetime import datetime

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

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
			import re
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

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

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

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

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
	import pandas as pd
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
	import pandas as pd
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
	
# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def QUBO_dimod(Q, beta):
	"""Creates a Binary Quadratic Model (BQM) from a Q matrix and an offset.
	This function constructs a BQM, which is used for solving quadratic optimization problems using binary variables.
	It uses the `dimod` library to create the model from the Q adjacency matrix and an offset value, preparing it for use with
	optimization solvers.
	Args:
		Q (np.ndarray): The Q matrix representing the quadratic terms in the optimization problem.
		beta (float): The offset value for the BQM.
	Returns:
		dimod.BinaryQuadraticModel: The Binary Quadratic Model constructed from the Q matrix and offset.
	Examples:
		>>> Q = np.array([[1, -1], [-1, 2]])
		>>> beta = 0.5
		>>> bqm = QUBO_dimod(Q, beta)
		>>> print(bqm)
		BinaryQuadraticModel({0: 1, 1: -1}, {(0, 1): -1}, 0.5, dimod.BINARY)
	"""
	bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=beta)
	return bqm

def build_Q_matrix(G, VC, EB, s, rho):
	"""Builds the Q matrix for the Quadratic Unconstrained Binary Optimization (QUBO) problem based on the given graph and model.
	This function constructs the Q adjacency matrix used in QUBO formulations, incorporating vertex costs, edge weights (edge betweenness), and constraints.
	Args:
		G (nx.Graph): The NetworkX graph object representing the water distribution network (WDN).
		VC (dict): Vertex cost dictionary for node costs.
		EB (dict): Edge betweenness dictionary for edge weights.
		s (int): The number of sensors to be placed.
	Returns:
		tuple: A tuple containing:
			- Q (np.ndarray): The Q matrix for the QUBO problem.
			- cQ (float): The constant term in the QUBO objective function.
	Raises:
		ValueError: If the model is None.
	Examples:
		>>> Q, cQ = build_Q_matrix(G, VC, EB, 5)
		>>> print(Q)
		[[ 1.2 -0.5  0. ]
		 [-0.5  1.5 -0.3]
		 [ 0.  -0.3  1.1]]
		>>> print(cQ)
		2.5
	"""
	nodes = list(G.nodes())
	vertex_cost = np.array([VC[node] for node in nodes])
	weight = EB
	num_nodes = len(nodes)
	A = np.ones((1, num_nodes))
	b = np.array([s])
	Q = np.diag(vertex_cost)
	total_weight = 0
	for (i, j), w in weight.items():
		if nodes.index(i) != nodes.index(j):
			Q[nodes.index(i), nodes.index(j)] += w/2
			Q[nodes.index(j), nodes.index(i)] += w/2
		Q[nodes.index(i), nodes.index(i)] -= w
		Q[nodes.index(j), nodes.index(j)] -= w
		total_weight += w
	Q += rho*np.matmul(A.T,A)
	Q -= rho*2*np.diag(np.matmul(b.T,A))
	cQ = rho * np.matmul(b.T, b) + total_weight
	return Q, cQ

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
	
def sensor_placement_results(model_MIQP, model_QUBO):
	"""Prints and returns sensor placement results for MIP and QUBO models.
	This function displays the sensor placement status for each node in both MIP and QUBO models. It provides
	a clear view of which nodes have sensors placed according to each optimization approach. It also returns
	dictionaries with sensor placement information for further analysis.
	Args:
		model_MIQP (pyo.ConcreteModel): The Pyomo model with MIP formulation, including sensor placement variables.
		model_QUBO (pyo.ConcreteModel): The Pyomo model with QUBO formulation, including sensor placement variables.
	Returns:
		tuple: A tuple containing two dictionaries:
			- sensor_placement_MIQP (dict): Dictionary with node names as keys and binary values (0 or 1) indicating
			  whether a sensor is placed at each node according to the MIP model.
			- sensor_placement_QUBO (dict): Dictionary with node names as keys and binary values (0 or 1) indicating
			  whether a sensor is placed at each node according to the QUBO model.
	Examples:
		>>> sensor_placement_MIQP, sensor_placement_QUBO = sensor_placement_results(model_MIQP, model_QUBO)
		>>> print(sensor_placement_MIQP)
		{'J1': 1, 'J2': 0, 'J3': 1}
		>>> print(sensor_placement_QUBO)
		{'J1': 1, 'J2': 1, 'J3': 0}
	"""
	print("MIP results")
	for node in model_MIQP.nodes:
		print(f"Node {node}: Sensor placed = {pyo.value(model_MIQP.x[node])}")
	sensor_placement_MIQP = {node: pyo.value(model_MIQP.x[node]) for node in model_MIQP.nodes}
	print("\nQUBO results")
	for node in model_QUBO.nodes:
		print(f"Node {node}: Sensor placed = {pyo.value(model_QUBO.x[node])}")
	sensor_placement_QUBO = {node: pyo.value(model_QUBO.x[node]) for node in model_QUBO.nodes}
	return sensor_placement_MIQP, sensor_placement_QUBO

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

def iterate_over_rho(s, rho_values):
    """
    Iterate over different rho values for the QUBO problem and find the objective value.
    """
    approx_errors_SA = []
    approx_errors_tabu = []

    for rho in rho_values:
        Q, cQ = build_Q_matrix(G, VC, EB, s, rho)
        bqm = QUBO_dimod(Q, beta=cQ) # returns bqm value

        # Run Simulated Annealing
        print("Running Simulated Annealing...")
        simAnnSampler = neal.SimulatedAnnealingSampler()
        simAnnSamples = simAnnSampler.sample(bqm, num_reads=1000)
        simAnnSamples.info
        # Find the minimum energy sample
        min_energy_sample = min(simAnnSamples.data(), key=lambda x: x.energy)
        print("Minimum energy sample:", min_energy_sample)

        # Extract the objective value from the samples
        objective_value_SA = min_energy_sample.energy
        approx_SA = calculate_approximation_ratio(objective_value_list_MIP_min, objective_value_SA)
        approx_errors_SA.append(approx_SA)

        # Run Tabu Search
        print("Running Tabu Search...")
        tabuSampler = TabuSampler()
        tabuSamples = tabuSampler.sample(bqm, num_reads=1000)
        tabuSamples.info
        # Find the minimum energy sample
        min_energy_sample_tabu = min(tabuSamples.data(), key=lambda x: x.energy)
        print("Minimum energy sample:", min_energy_sample_tabu)

        # Extract the objective value from the samples
        objective_value_tabu = min_energy_sample_tabu.energy
        approx_Tabu = calculate_approximation_ratio(objective_value_list_MIP_min, objective_value_tabu)
        approx_errors_tabu.append(approx_Tabu)

    return approx_errors_SA, approx_errors_tabu

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
	
def coverage(G, s, model, MIQP, solver):
	"""Computes and plots coverage metrics for the Water Distribution Network (WDN) using both MIP and QUBO formulations.
	This function iteratively solves the MIP and QUBO models for different numbers of sensors and records
	the objective values. It uses the Gurobi solver to find the optimal solutions and appends the results
	to lists for comparison.
	Args:
		G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
		s (list): The list of sensor counts to try.
		rho (float): Penalty parameter for the QUBO model.
		model: The base Pyomo model to clone for each run.
		MIQP: The MIQP function to use for model construction.
		solver: The Pyomo solver instance to use.
	Returns:
		list: A list of objective values obtained from the MIP formulation.
	Raises:
		ValueError: If the solver or model is not properly defined.
	Examples:
		>>> objective_value_list_MIQP = coverage(G, s, rho, model, MIQP, solver)
		>>> print(objective_value_list_MIQP)
		[100, 95, 90]
	"""
	objective_value_list_MIP = []
	for s_val in s:
		model_MIQP = MIQP(G, model, s_val)
		results_MIQP = solver.solve(model_MIQP, tee=True)
		objective_value_MIP = pyo.value(model_MIQP.obj)
		objective_value_list_MIP.append(objective_value_MIP)
	return objective_value_list_MIP

def QUBO(G, model_in, s, rho):
	"""Generates a Quadratic Unconstrained Binary Optimization (QUBO) model for the given problem.
	This function defines the objective function for the QUBO model, which aims to optimize sensor placement
	in the city's Water Distribution Network (WDN). The objective function includes terms for vertex costs,
	edge interactions, and a penalty for deviating from the specified number of sensors.
	Args:
		G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
		model_in (pyo.ConcreteModel): A Pyomo ConcreteModel instance with sets and parameters defined.
		s (int): The number of sensors to be placed in the network.
		rho (float): Penalty parameter for the constraint that exactly `s` sensors must be placed.
	Returns:
		pyo.ConcreteModel: The updated Pyomo model with the QUBO objective function added.
	Raises:
		ValueError: If the model is None.
	Examples:
		>>> model = QUBO(G, model, 5, 0.1)
		>>> print(model.obj)
	"""
	if model_in is None:
		return None
	model = model_in.clone()
	def objective_rule_QUBO(model):
		term1 = sum(model.c[i] * model.x[i] for i in model.nodes)
		term2 = sum(model.w[(i, j)] * (1 - model.x[i]) * (1 - model.x[j]) for (i, j) in model.edges)
		term3 = rho * (sum(model.x[i] for i in model.nodes) - s) ** 2
		return term1 + term2 + term3
	model.obj = pyo.Objective(rule=objective_rule_QUBO, sense=pyo.minimize)
	return model

def MIQP(G, model_in, s):
	"""Generates a Mixed Integer Programming (MIP) model for the given sensor placement problem.
	This function defines the objective function and constraints for the MIP model, aiming to optimize
	sensor placement in the city's Water Distribution Network (WDN). The objective function minimizes
	the total cost based on vertex costs and edge betweenness, while the constraint ensures that exactly
	`s` sensors are placed.
	Args:
		G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
		model_in (pyo.ConcreteModel): A Pyomo ConcreteModel instance with sets and parameters defined.
		s (int): The number of sensors to be placed in the network.
	Returns:
		pyo.ConcreteModel: The updated Pyomo model with the objective function and constraints added.
	Raises:
		ValueError: If the model is None.
	Examples:
		>>> model = MIQP(G, model, 5)
		>>> print(model.obj)
	"""
	if model_in is None:
		return None
	model = model_in.clone()
	def objective_rule_MIQP(model):
		return sum(model.c[i] * model.x[i] for i in model.nodes) + \
				sum(model.w[(i, j)] * (1 - model.x[i] - model.x[j] + model.x[i] * model.x[j]) for (i, j) in model.edges)
	model.obj = pyo.Objective(rule=objective_rule_MIQP, sense=pyo.minimize)
	def sensor_constraint_rule_MIQP(model):
		return sum(model.x[i] for i in model.nodes) <= s
	model.sensor_constraint_MIQP = pyo.Constraint(rule=sensor_constraint_rule_MIQP)
	return model

def create_pyomo_model(G, water_consumption, VC, EB, city):
	"""Initializes a Pyomo model for optimizing sensor placement in a city's Water Distribution Network (WDN).
	This function sets up a Pyomo model with nodes, edges, and associated parameters such as demand,
	vertex cost, and edge betweenness. It also defines the binary decision variables for sensor placement.
	Args:
		G (nx.Graph): A NetworkX graph object representing the water distribution network (WDN) for the city.
		water_consumption (dict): Water consumption dictionary for demand parameter.
		VC (dict): Vertex cost dictionary for node costs.
		EB (dict): Edge betweenness dictionary for edge weights.
		city (str): Name of the city (for error messages/logging).
	Returns:
		pyo.ConcreteModel: A Pyomo ConcreteModel instance with sets, parameters, and decision variables defined.
	Raises:
		ValueError: If the graph `G` is None.
	Examples:
		>>> model = create_pyomo_model(G, water_consumption, VC, EB, city)
		>>> print(model)
	"""
	if G is None:
		print(f"{city} does not exist.")
		return None
	model = pyo.ConcreteModel()
	nodes = list(G.nodes())
	edges = list(G.edges())
	model.nodes = pyo.Set(initialize=nodes)
	model.edges = pyo.Set(initialize=edges, dimen=2)
	demand = water_consumption
	model.demand = pyo.Param(model.nodes, initialize=demand, mutable=True)
	model.c = pyo.Param(model.nodes, initialize=VC, mutable=True)
	model.w = pyo.Param(model.edges, initialize=EB, mutable=True)
	model.x = pyo.Var(model.nodes, within=pyo.Binary)
	return model

# ============================================================================
# NETWORK/GRAPH FUNCTIONS
# ============================================================================

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
	
def centrality(G, water_consumption):
	"""Calculates centrality metrics for a given city's WDN graph and water consumption data.
	This function computes various centrality metrics for the provided NetworkX graph and calculates
	vertex costs based on water consumption data and centrality measures.
	Args:
		G (nx.Graph): A NetworkX graph representing the water distribution network (WDN) for the city.
		water_consumption (dict): A dictionary where keys are node names and values are the water consumption
			at each node.
	Returns:
		tuple: A tuple containing two elements:
			- vertex_cost (dict): Dictionary where keys are node names and values are the computed costs
			  for each vertex based on normalized demand and degree centrality.
			- edge_betweenness (dict): Dictionary of edge betweenness centrality values, where keys are
			  tuples representing edges and values are the betweenness centrality scores for those edges.
	Raises:
		KeyError: If a node from the graph is not found in the water consumption data.
	Examples:
		>>> vertex_cost, edge_betweenness = centrality(G, water_consumption)
		>>> print(vertex_cost)
		{'J1': 2.5, 'J2': 3.0}
		>>> print(edge_betweenness)
		{(J1, J2): 0.1, (J2, J3): 0.2}
	"""

	# Calculating edge betweenness centrality
	edge_betweenness = nx.edge_betweenness_centrality(G)
	# Calculate degree centrality
	degree_centrality = nx.degree_centrality(G)
	# Edge betweenness centrality as weights
	for (u, v, d) in G.edges(data=True):
		d['weight'] = edge_betweenness[(u, v)]
	# Define the weights C and D (Need to identify the right values)
	C = 1.0
	D = 1.0
	def compute_normalized_demand(water_consumption):
		max_demand = max(water_consumption.items(), key=operator.itemgetter(1))[1]
		return {node: val / max_demand for node, val in water_consumption.items()}
	normalized_demand = compute_normalized_demand(water_consumption)
	# Vertex cost function:
	vertex_cost = {}
	for node in G.nodes():
		f_i = normalized_demand.get(node, None)   # function of the water need at each node i
		if f_i is None:
			continue
		g_i = degree_centrality[node] / (len(G.nodes()) - 1)    # node weights
		vertex_cost[node] = C * f_i + D * g_i
	return vertex_cost, edge_betweenness

def WDN_network_data(city, base_dir=None):
	"""Fetches and processes Water Distribution Network (WDN) data for a given city.
	This function reads node, edge, and water consumption data from specified files
	for a given city and returns the data as dictionaries.
	Args:
		city (str): The name of the city to fetch data for. Should be one of 'Apulia', 
			'Fossolo', or 'Test'.
		base_dir (str, optional): The base directory where the data files are located. 
			Defaults to the directory of the script.
	Returns:
		tuple: A tuple containing three dictionaries:
			- nodes (dict): Dictionary of nodes with node names as keys and (x, y) 
			  coordinates as values.
			- edges (dict): Dictionary of edges with tuples of node pairs as keys.
			- water_consumption (dict): Dictionary of water consumption with node names
			  as keys and consumption values as values.
	Raises:
		FileNotFoundError: If any of the data files do not exist.
		ValueError: If the provided city name is not valid.
	Examples:
		>>> nodes, edges, water_consumption = WDN_network_data('Apulian')
		>>> print(nodes)
		{'J1': (10.0, 20.0), 'J2': (15.0, 25.0)}
	"""
	# Set base directory
	if base_dir is None:
		base_dir = os.path.dirname(os.path.abspath(__file__))  # Set to script's directory
	data_dir = os.path.join(base_dir, 'Data/WDN_Data')
	city_files = {
		"Apulia": {
			"nodes": f"{data_dir}/Apulia_WDN/Apulia_WDN_Nodes.txt",
			"edges": f"{data_dir}/Apulia_WDN/Apulia_WDN_Edges.txt",
			"water_consumption": f"{data_dir}/Apulia_WDN/Apulia_WDN_Water_Consumption.txt"
		},
		"Fossolo": {
			"nodes": f"{data_dir}/Fossolo_WDN/Fossolo_WDN_Nodes.txt",
			"edges": f"{data_dir}/Fossolo_WDN/Fossolo_WDN_Edges.txt",
			"water_consumption": f"{data_dir}/Fossolo_WDN/Fossolo_WDN_Water_Consumption.txt"
		},
		"Test": {
			"nodes": f"{data_dir}/Test_WDN/Test_Nodes.txt",
			"edges": f"{data_dir}/Test_WDN/Test_Edges.txt",
			"water_consumption": f"{data_dir}/Test_WDN/Test_Water_Consumption.txt"
		},
		"ZJ": {
			"nodes": f"{data_dir}/ZJ_WDN/ZJ_WDN_Nodes.txt",
			"edges": f"{data_dir}/ZJ_WDN/ZJ_WDN_Edges.txt",
			"water_consumption": f"{data_dir}/ZJ_WDN/ZJ_WDN_Water_Consumption.txt"
		},
		"Modena": {
			"nodes": f"{data_dir}/Modena_WDN/Modena_WDN_Nodes.txt",
			"edges": f"{data_dir}/Modena_WDN/Modena_WDN_Edges.txt",
			"water_consumption": f"{data_dir}/Modena_WDN/Modena_WDN_Water_Consumption.txt"
		},
		"Kentucky": {
			"nodes": f"{data_dir}/Kentucky_WDN/Kentucky_WDN_Nodes.txt",
			"edges": f"{data_dir}/Kentucky_WDN/Kentucky_WDN_Edges.txt",
			"water_consumption": f"{data_dir}/Kentucky_WDN/Kentucky_WDN_Water_Consumption.txt"
		}
	}
	if city not in city_files:
		print(f"Error: Unknown city {city}.")
		return None, None, None
	nodes_file_path = city_files[city]["nodes"]
	edges_file_path = city_files[city]["edges"]
	water_consumption_file_path = city_files[city]["water_consumption"]
	print(f"Nodes file path: {nodes_file_path}")
	print(f"Edges file path: {edges_file_path}")
	print(f"Water consumption file path: {water_consumption_file_path}")
	for file_path in [nodes_file_path, edges_file_path, water_consumption_file_path]:
		if not os.path.exists(file_path):
			print(f"Error: The file {file_path} does not exist.")
			return None, None, None
	nodes_df = pd.read_csv(nodes_file_path, skiprows=1, names=["Node", "X", "Y"])
	nodes_df['Node'] = nodes_df['Node'].str.strip()
	nodes = {row['Node']: (float(row['X']), float(row['Y'])) for i, row in nodes_df.iterrows()}
	edges_df = pd.read_csv(edges_file_path, skiprows=1, names=["Node1", "Node2"])
	edges_df['Node1'] = edges_df['Node1'].str.strip()
	edges_df['Node2'] = edges_df['Node2'].str.strip()
	edges = {((row['Node1']), (row['Node2'])) for i, row in edges_df.iterrows()}
	water_consumption_df = pd.read_csv(water_consumption_file_path, skiprows=1, names=["Node", "Consumption"])
	water_consumption_df['Node'] = water_consumption_df['Node'].str.strip()
	water_consumption = {row['Node']: (float(row['Consumption'])) for i, row in water_consumption_df.iterrows()}
	return nodes, edges, water_consumption

def Construct_Graph(city, coords, edges):

    """Constructs a Water Distribution Network (WDN) graph for a given city.

    This function constructs a graph using the node coordinates and edges imported previously.
    It creates a NetworkX graph, adds nodes with their coordinates as attributes, and adds edges.

    Args:
        city (str): The name of the city for which to construct the graph. The function assumes
            that the node coordinates and edges for the specified city have been imported.

    Returns:
        nx.Graph: A NetworkX graph object representing the WDN. If the coordinates or edges are
            not available, returns None.

    Raises:
        ValueError: If the city name is not valid or if the necessary data is missing.

    Examples:
        >>> G = Construct_Graph('Apulian')
        >>> print(G.nodes(data=True))
        [('J1', {'pos': (10.0, 20.0)}), ('J2', {'pos': (15.0, 25.0)})]
    """

    if coords is None or edges is None:
        return None
    
    G = nx.Graph()

    # Defining Nodes in their coordinates
    for node, coord in coords.items():
        G.add_node(node, pos=coord)

    # Add edges
    G.add_edges_from(edges)
    
    return G
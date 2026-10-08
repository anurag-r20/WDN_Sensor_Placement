"""Reading water distribution network (WDN) data and turning it into a scored graph.

Pipeline position: network files -> ``WDN_network_data`` -> ``Construct_Graph`` -> ``centrality``
-> vertex costs ``VC`` and edge weights ``EB`` used by :mod:`src.formulations`.
"""

import operator
import os

import networkx as nx
import pandas as pd


def WDN_network_data(city, base_dir=None):
    """Fetches and processes Water Distribution Network (WDN) data for a given city.
    This function reads node, edge, and water consumption data from specified files
    for a given city and returns the data as dictionaries.
    Args:
        city (str): The name of the city to fetch data for. Should be one of 'Apulia', 
            'Fossolo', or 'Test'.
        base_dir (str, optional): The base directory where the data files are located. 
            Defaults to the repository root (the folder that contains src/).
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Repository root
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

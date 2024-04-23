import numpy as np
import scipy as sp
import scipy.sparse as sps

from collections import Counter


def to_one_hot(y, n_classes):
    """
    Transforms a vector of integers into a one-hot encoding
    """
    return np.eye(n_classes)[y]


def read_dataset(
    DS,
    include_node_attributes=False,
    include_node_labels=True,
    nodes_padding=0,
    edges_padding=0,
):
    """
    Reads a graph formatted multifile dataset with the following structure:
    (replace DS by the name of the dataset):

    n = total number of nodes
    m = total number of edges
    N = number of graphs

    (1) 	DS_A.txt (m lines)
            sparse (block diagonal) adjacency matrix for all graphs,
            each line corresponds to (row, col) resp. (node_id, node_id)

    (2) 	DS_graph_indicator.txt (n lines)
            column vector of graph identifiers for all nodes of all graphs,
            the value in the i-th line is the graph_id of the node with node_id i

    (3) 	DS_graph_labels.txt (N lines)
            class labels for all graphs in the dataset,
            the value in the i-th line is the class label of the graph with graph_id i

    (4) 	DS_node_labels.txt (n lines)
            column vector of node labels,
            the value in the i-th line corresponds to the node with node_id i

    There are OPTIONAL files if the respective information is available:

    (5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
            labels for the edges in DS_A_sparse.txt

    (6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
            attributes for the edges in DS_A.txt

    (7) 	DS_node_attributes.txt (n lines)
            matrix of node attributes,
            the comma seperated values in the i-th line is the attribute vector of the node with node_id i

    (8) 	DS_graph_attributes.txt (N lines)
            regression values for all graphs in the dataset,
            the value in the i-th line is the attribute of the graph with graph_id i
    """

    # read the graph labels
    graph_labels = np.loadtxt(f"{DS}_graph_labels.txt", dtype=int)
    graph_labels_one_hot = to_one_hot(graph_labels, np.max(graph_labels) + 1)

    # read the graph indicator
    graph_indicator = np.loadtxt(f"{DS}_graph_indicator.txt", dtype=int).tolist()

    node_global_to_local = {}
    graph_node_counter = {graph_id: 1 for graph_id in range(1, len(graph_labels) + 1)}

    if include_node_labels:
        node_labels = np.loadtxt(f"{DS}_node_labels.txt", dtype=int)
        node_labels = to_one_hot(node_labels, np.max(node_labels) + 1)

    for node in range(1, len(graph_indicator) + 1):
        graph_id = graph_indicator[node - 1]
        node_global_to_local[node] = graph_node_counter[graph_id]
        graph_node_counter[graph_id] += 1

    # read the edge list
    with open(f"{DS}_A.txt") as f:
        edge_list = f.readlines()
        edge_list = [list(map(int, edge.split(","))) for edge in edge_list]

    # find the number of max nodes and edges in a graph
    max_nodes = (
        max(graph_node_counter.values()) + nodes_padding + 1
    )  # +1 for the dummy 0 node
    max_edges = (
        max(Counter(map(lambda x: graph_indicator[x[0] - 1], edge_list)).values())
    ) * 2 + edges_padding

    print(f"Max nodes: {max_nodes}, Max edges: {max_edges}")

    if include_node_attributes:
        node_attributes = np.loadtxt(f"{DS}_node_attributes.txt", delimiter=",")

        if include_node_labels:
            node_labels = np.concatenate([node_labels, node_attributes], axis=1)
        else:
            node_labels = node_attributes

    # split all the data based on the graph belonging
    node_features_split = np.zeros((len(graph_labels), max_nodes, node_labels.shape[1]))
    edge_list_split = np.zeros((len(graph_labels), max_edges, 2), dtype=int)
    edge_list_split_add_indices = [0] * len(graph_labels)
    degrees_split = np.zeros((len(graph_labels), max_nodes))

    for node in range(1, len(graph_indicator) + 1):
        graph_id = graph_indicator[node - 1]
        local_node = node_global_to_local[node]
        node_features_split[graph_id - 1, local_node] = node_labels[node - 1]

    for edge in edge_list:
        graph_id = graph_indicator[edge[0] - 1]
        local_node_1 = node_global_to_local[edge[0]]
        local_node_2 = node_global_to_local[edge[1]]
        edge_list_split[graph_id - 1, edge_list_split_add_indices[graph_id - 1]] = [
            local_node_1,
            local_node_2,
        ]
        degrees_split[graph_id - 1, local_node_1] += 1
        edge_list_split_add_indices[graph_id - 1] += 1

    return node_features_split, edge_list_split, degrees_split, graph_labels_one_hot

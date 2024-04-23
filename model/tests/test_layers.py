import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import pytest
import numpy as np
import random

from keras import ops

from model.layers.attention import SingleHeadAttention
from model.layers.common_layers import GatherNodes, ReduceGatheredNodesSum
from model.layers.convolution import SingleGraphConvolution


@pytest.fixture
def bull_graph():
    # the bull graph, see https://en.wikipedia.org/wiki/Bull_graph
    num_nodes = 5

    edge_list = [[0, 1], [1, 2], [1, 3], [2, 3], [3, 4]]

    # make the graph undirected by adding the reverse edges
    edge_list += [[edge[1], edge[0]] for edge in edge_list]

    edge_list = np.array(edge_list, dtype=np.int32)

    node_features = np.random.rand(num_nodes, 42)

    degrees = np.array([1, 2, 2, 2, 1], dtype=np.int32)

    return node_features, edge_list, degrees


@pytest.fixture
def butterfly_graph():
    # the butterfly graph, see https://en.wikipedia.org/wiki/Butterfly_graph
    num_nodes = 5

    edge_list = [
        [0, 1],
        [0, 2],
        [1, 2],
        [2, 3],
        [2, 4],
        [3, 4],
    ]

    # make the graph undirected by adding the reverse edges
    edge_list += [[edge[1], edge[0]] for edge in edge_list]

    edge_list = np.array(edge_list, dtype=np.int32)

    node_features = np.random.rand(num_nodes, 42)

    degrees = np.array([2, 2, 4, 2, 2], dtype=np.int32)

    return node_features, edge_list, degrees


def random_graph_undirected():
    num_nodes = random.randint(10, 100)

    # generate the edges
    edge_list = []

    for _ in range(random.randint(num_nodes, num_nodes * 2)):
        edge = [random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)]
        if edge[0] != edge[1] and edge not in edge_list and edge[::-1] not in edge_list:
            edge_list.append(edge)

    # make the graph undirected by adding the reverse edges
    edge_list += [[edge[1], edge[0]] for edge in edge_list]

    edge_list = np.array(edge_list, dtype=np.int32)

    node_features = np.random.rand(num_nodes, 42)

    degrees = np.array(
        [
            len([edge for edge in edge_list if node == edge[0]])
            for node in range(num_nodes)
        ],
        dtype=np.int32,
    )

    return node_features, edge_list, degrees


def test_gather_nodes():
    """
    Tests the GatherNodes layer on a random graph
    """
    node_features, edge_list, degrees = random_graph_undirected()

    num_nodes = node_features.shape[0]
    num_features = node_features.shape[1]
    num_edges = edge_list.shape[0]

    gathered_nodes = GatherNodes()([node_features, ops.cast(edge_list, "int32")])

    # check the shape
    assert gathered_nodes.shape == (num_edges, 2, num_features)

    # check the values
    for i, edge in enumerate(edge_list):
        for j, node in enumerate(edge):
            assert np.allclose(gathered_nodes[i, j], node_features[node], atol=1e-5)


def test_reduce_gathered_nodes():
    """
    Tests the ReduceGatheredNodes layer (and also the GatherNodes layer) on a random graph
    """
    node_features, edge_list, degrees = random_graph_undirected()

    gathered_nodes = GatherNodes()([node_features, ops.cast(edge_list, "int32")])

    reduced_nodes = ReduceGatheredNodesSum()(
        [node_features, gathered_nodes, ops.cast(edge_list, "int32")]
    )

    # check the shape
    assert reduced_nodes.shape == node_features.shape

    # check the values
    true_results = np.zeros_like(node_features)

    for u, v in edge_list:
        true_results[u] += node_features[v]

    assert np.allclose(reduced_nodes, true_results, atol=1e-5)


def test_convolution():
    """
    Tests the GraphConvolution layer on a random graph
    """

    node_features, edge_list, degrees = random_graph_undirected()

    num_nodes = node_features.shape[0]
    num_features = node_features.shape[1]
    num_edges = edge_list.shape[0]

    output_features = random.randint(32, 256)

    convolution_layer = SingleGraphConvolution(output_features, activation=None)
    convolution_output = convolution_layer(
        [node_features, ops.cast(edge_list, "int32"), degrees]
    )

    # check the shape
    assert convolution_output.shape == (num_nodes, output_features)

    # check the values by manually applying the convolution
    kernel = convolution_layer.kernel.numpy()
    bias = convolution_layer.bias.numpy()

    convolution_truth = np.zeros_like(node_features)

    for u, v in edge_list:
        convolution_truth[u] += node_features[v] / np.sqrt(
            max(degrees[u] * degrees[v], 1)
        )

    # add the self loops to the ground truth
    convolution_truth += node_features / np.expand_dims(
        np.maximum(degrees, 1.0), axis=-1
    )

    convolution_truth = convolution_truth @ kernel + bias

    assert np.allclose(convolution_output, convolution_truth, atol=1e-5)


def test_single_head_attention():
    """
    Tests the SingleHeadAttention layer on a random graph
    """

    node_features, edge_list, degrees = random_graph_undirected()

    num_nodes = node_features.shape[0]
    num_features = node_features.shape[1]
    num_edges = edge_list.shape[0]

    output_features = random.randint(32, 256)

    attention_layer = SingleHeadAttention(output_features, activation=None)

    attention_output = attention_layer([node_features, ops.cast(edge_list, "int32")])

    # check the shape
    assert attention_output.shape == (num_nodes, output_features)

    # check the values by manually applying the attention
    W = attention_layer.kernel.numpy().T
    b = attention_layer.bias.numpy()
    a = attention_layer.attention_kernel.numpy()

    attention_coefs = np.zeros((num_nodes, num_nodes))

    for u, v in edge_list:
        attention_coefs[u, v] = np.exp(
            a.T @ np.concatenate([W @ node_features[u], W @ node_features[v]])
        ).item()

    for u in range(num_nodes):
        attention_coefs[u] /= np.sum(attention_coefs[u])

    attention_coefs = np.nan_to_num(attention_coefs)

    attention_truth = np.zeros((num_nodes, output_features))

    for u, v in edge_list:
        attention_truth[u] += attention_coefs[u, v] * W @ node_features[v]

    for u in range(num_nodes):
        if degrees[u] > 0:
            assert np.isclose(1.0, np.sum(attention_coefs[u]))
        else:
            assert np.allclose(attention_coefs[u], 0.0)

    assert np.allclose(attention_output, attention_truth, atol=1e-5)

# Graph Neural Networks for Graph Classification in Keras

This repository contains layers helpful for building Graph Neural Networks (GNNs). The layers are implemented using only Keras backend-agnostic operations, which means they can be used with any backend supported by Keras (currently, this includes PyTorch, Tensorflow and JAX).

The 2 key features of this repository are:
- Graph Convolution layer, as per Kipf et al https://arxiv.org/abs/1609.02907
- Graph Attention layer, as per Velickovic et al https://arxiv.org/abs/1710.10903

The layers are stand-alone and can be used and extended to any Keras model. The repository also contains examples on how to use them.

## Data format

The advantage of this repository is its memory efficiency, and the way it deals with batches.

A graph is represented by 2 (potentially 3) matrices:
- Node feature matrix: A matrix of shape `(num_nodes, num_features)`, where `num_nodes` is the number of nodes in the graph, and `num_features` is the number of features per node. This matrix represents the features of each node in the graph. **Important**: The first row of this matrix should be all zero, as it is the dummy padding node. Thus, the actual nodes start from the index 1. Moreover, the matrix should be padded with zeros to have a fixed number of nodes per graph. Thus, the final shape of the matrix should be `(max_num_nodes + 1, num_features)`, where `max_num_nodes` is the maximum number of nodes in the graph, and `+1` comes from the `0th` dummy node.
- Edge list: A matrix of shape `(num_edges, 2)`, where `num_edges` is the number of edges in the graph. Each row of this matrix represents an edge in the graph. The first column is the source node, and the second column is the destination node. **Important**: The source and destination nodes should be 1-indexed, as the 0th index is reserved for the dummy padding node. Once again, the matrix should be padded with zeros to have a fixed number of edges per graph. Thus, the final shape of the matrix should be `(max_num_edges, 2)`.
- (Optional) Degrees matrix: A matrix of shape `(max_num_nodes + 1, 1)`, used only by the `GraphConvolution` layer, where `degrees[i]` is equal to the degree of the correspong vertex represented at `node_features[i]`.

The advantage of this format is that the computation needs only `O(V * F + E * F)` memory, where `V` is the number of nodes, `F` is the number of features per node, and `E` is the number of edges. This is in contrast to the adjacency matrix format (used by PyTorch geometric), which requires at least `O(V^2)` memory.

Consequently, a batch of graphs is created by simply stacking the node feature matrices and edge lists along the first axis. The degrees matrix is also stacked along the first axis. This again comes in contrast to the method in which a batch is created by joining multiple graphs into a single, disconnected graph, as the latter method requires more padding, and thus memory.

### Loading data

To help with data loading, a function `read_dataset()` is provided, which builds a corresponding dataset from a list of files in the **TUDataset** format (see https://chrsmrrs.github.io/datasets/docs/format/).



import keras
import numpy as np

from keras import ops
from keras import layers

from model.layers.common_layers import GatherNodes, ReduceGatheredNodesSum


class SingleGraphConvolution(layers.Layer):
    """
    Graph convolution layer over a single graph, as per https://arxiv.org/abs/1609.02907
    """

    def __init__(self, units, activation=ops.relu, use_bias=True, **kwargs):
        super(SingleGraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.activation = activation
        self._gather_layer = GatherNodes()
        self._reduce_layer = ReduceGatheredNodesSum()

    def build(self, input_shape):
        super(SingleGraphConvolution, self).build(input_shape)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[0][-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
            )

    def compute_output_shape(self, inputs_shape):
        output_shape = list(inputs_shape[0])
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = super(SingleGraphConvolution, self).get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config

    def call(self, inputs):
        """
        Graph convolution layer

        Args:
            inputs: a list of tensors:
                - the first tensor is the node features of shape (num_nodes, num_features)
                - the second tensor is the edge list of shape (num_edges, 2)
                - the third tensor is the degrees of the nodes of shape (num_nodes,)

        Returns: a tensor of shape (num_nodes, num_features)
        """
        node_features, edge_list, degrees = inputs

        # make degrees of shape (num_nodes, 1)
        # degrees = ops.expand_dims(degrees, axis=-1)

        # multiply the features of the nodes with the rsqrt of their degrees

        # edge_list[i] = [u, v]
        # gathered_nodes[i] = [node_features[u], node_features[v]]
        # gathered_degrees[i] = [degree[u], degree[v]]
        gathered_node_features = self._gather_layer([node_features, edge_list])

        gathered_degrees = self._gather_layer([degrees, edge_list])

        # reduce the product of the gathered degrees, so that
        # gathered_degrees[i] = degree[u] * degree[v]
        gathered_degrees = ops.prod(gathered_degrees, axis=-1)

        # compute the inverse of the product of the degrees
        # gathered_degrees[i] = 1 / sqrt(degree[u] * degree[v])
        gathered_degrees = ops.rsqrt(ops.maximum(gathered_degrees, 1.0))
        gathered_degrees = ops.expand_dims(gathered_degrees, axis=-1)

        # multiply the features of the nodes with the rsqrt of their degrees
        gathered_node_features = ops.multiply(
            gathered_node_features, ops.expand_dims(gathered_degrees, axis=-1)
        )

        gathered_nodes = self._reduce_layer(
            [node_features, gathered_node_features, edge_list]
        )

        # add the self loop to the gathered nodes
        gathered_nodes = ops.add(
            gathered_nodes,
            ops.divide(
                node_features, ops.expand_dims(ops.maximum(degrees, 1.0), axis=-1)
            ),
        )

        # apply the convolution
        gathered_nodes = ops.matmul(gathered_nodes, self.kernel)

        if self.use_bias:
            gathered_nodes = ops.add(gathered_nodes, self.bias)

        if self.activation is not None:
            gathered_nodes = self.activation(gathered_nodes)

        return gathered_nodes

    def get_config(self):
        config = super(SingleGraphConvolution, self).get_config()
        config.update(
            {
                "units": self.units,
                "activation": keras.saving.serialize_keras_object(self.activation),
                "use_bias": self.use_bias,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        activation = config.pop("activation")
        activation = keras.saving.deserialize_keras_object(activation)
        return cls(activation=activation, **config)

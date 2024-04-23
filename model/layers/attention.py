import keras
import numpy as np

from keras import ops
from keras import layers

from model.layers.common_layers import GatherNodes, ReduceGatheredNodesSum


class SingleHeadAttention(layers.Layer):
    """
    GAT as defined in https://arxiv.org/abs/1710.10903
    """

    def __init__(self, output_dim, activation=ops.relu, use_bias=True, **kwargs):
        super(SingleHeadAttention, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self._gather_layer = GatherNodes()
        self._reduce_layer = ReduceGatheredNodesSum()

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[0][-1], self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.attention_kernel = self.add_weight(
            name="attention_kernel",
            shape=(
                2 * self.output_dim,
                1,
            ),
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.output_dim,),
                initializer="zeros",
                trainable=True,
            )

    def compute_output_shape(self, inputs_shape):
        inputs_shape = list(inputs_shape[0])
        inputs_shape[-1] = self.output_dim
        return tuple(inputs_shape)

    def call(self, inputs):
        """
        Graph attention layer, see https://arxiv.org/abs/1710.10903

        Args:
            inputs: a list of tensors:
                - the first tensor is the node features of shape (num_nodes, num_features)
                - the second tensor is the edge list of shape (num_edges, 2)
        Returns: a tensor of shape (num_nodes, num_features)
        """

        node_features, edge_list = inputs
        num_nodes = node_features.shape[0]

        # compute the attention keys
        attention_keys = ops.matmul(node_features, self.kernel)

        # edge_list[i] = [u, v]
        # gathered_nodes[i] = [attention_keys[u], attention_keys[v]]
        gathered_nodes = self._gather_layer([attention_keys, edge_list])

        # reduce the gathered nodes by concatenating them
        gathered_nodes_reduced = ops.reshape(gathered_nodes, (-1, 2 * self.output_dim))
        # apply the attention kernel
        # attention[i] = attention_head[u] & attention_head[v], where & is the concatenation
        attention_mask = ops.matmul(gathered_nodes_reduced, self.attention_kernel)

        # apply the exponential
        attention_mask = ops.exp(attention_mask)

        # sum the attentions over the neighborhood of each node

        outgoing_indices = edge_list[:, 0]
        outgoing_indices = ops.expand_dims(outgoing_indices, axis=-1)

        # attention_neighborhood[i] = sum_{attention[j] | j in N(i)}
        attention_neighborhood = ops.scatter(
            outgoing_indices, attention_mask, (num_nodes, 1)
        )

        # attention_neighborhood[i] = sum_{attention[x] | x in N(u)}
        attention_neighborhood_gathered = ops.take(
            attention_neighborhood, edge_list[:, 0], axis=0
        )

        # normalize the attention
        attention_coeffs_list = ops.divide_no_nan(
            attention_mask, attention_neighborhood_gathered
        )

        attention_coeffs = ops.scatter(
            edge_list,
            attention_coeffs_list,
            (num_nodes, num_nodes, 1),
        )

        attention_coeffs = ops.squeeze(attention_coeffs, axis=-1)

        # finally apply the attention
        # first, we get the corresponding indices of the attention coefficients
        # to do this, we need the flat indices of the attention coefficients
        flat_indices = ops.multiply(edge_list[:, 1], num_nodes) + edge_list[:, 0]

        # then we gather the attention coefficients
        attention_coeffs = ops.take(attention_coeffs, flat_indices)

        # apply the attention
        gathered_nodes = ops.multiply(
            gathered_nodes, ops.reshape(attention_coeffs, (-1, 1, 1))
        )

        # reduce the gathered nodes
        gathered_nodes = self._reduce_layer(
            [ops.empty((num_nodes, self.output_dim)), gathered_nodes, edge_list]
        )

        if self.use_bias:
            gathered_nodes = ops.add(gathered_nodes, self.bias)

        if self.activation is not None:
            gathered_nodes = self.activation(gathered_nodes)

        return gathered_nodes

    def get_config(self):
        config = super(SingleHeadAttention, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
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


class MultiHeadAttention(layers.Layer):
    """
    Applies multiple single head attentions on a graph and concatenates the results
    """

    def __init__(
        self, output_dim, num_heads, activation=ops.relu, use_bias=True, **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.activation = activation
        self.use_bias = use_bias
        self.built = False

        assert self.output_dim % self.num_heads == 0

        self.head_dim = self.output_dim // self.num_heads

        self.heads = [
            SingleHeadAttention(self.head_dim, activation, use_bias)
            for _ in range(self.num_heads)
        ]

    def build(self, input_shape):
        if not self.built:
            super(MultiHeadAttention, self).build(input_shape)

            # build the children layers
            for head in self.heads:
                head.build(input_shape)

            self.built = True

    def compute_output_shape(self, inputs_shape):
        inputs_shape = list(inputs_shape[0])
        inputs_shape[-1] = self.output_dim
        return tuple(inputs_shape)

    def call(self, inputs):
        """
        Concatenates the results of multiple attention heads

        Args:
            inputs: a list of tensors:
                - the first tensor is the node features of shape (num_nodes, num_features)
                - the second tensor is the edge list of shape (num_edges, 2)

        Returns: a tensor of shape (num_nodes, num_features)
        """

        node_features, edge_list = inputs

        # apply the attention heads
        heads = [head([node_features, edge_list]) for head in self.heads]

        # concatenate the results
        return ops.concatenate(heads, axis=-1)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_heads": self.num_heads,
                "activation": keras.saving.serialize_keras_object(self.activation),
                "use_bias": self.use_bias,
                "heads": [
                    keras.saving.serialize_keras_object(head) for head in self.heads
                ],
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        activation = config.pop("activation")
        activation = keras.saving.deserialize_keras_object(activation)
        heads = [
            keras.saving.deserialize_keras_object(head) for head in config.pop("heads")
        ]

        instance = cls(activation=activation, **config)

        instance.heads = heads
        instance.built = True

        return instance

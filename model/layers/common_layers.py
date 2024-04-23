import keras
import numpy as np

from keras import ops
from keras import layers


class GatherNodes(layers.Layer):
    """
    Gathers the features of the nodes associated with each edge
    """

    def __init__(self, **kwargs):
        super(GatherNodes, self).__init__(**kwargs)

    def compute_output_shape(self, inputs_shape):
        node_features_shape, edge_list_shape = inputs_shape
        return (edge_list_shape[0], edge_list_shape[1], node_features_shape[-1])

    def call(self, inputs):
        """
        Args:
            inputs: a list of two tensors, the first tensor is the node features of shape (num_nodes, num_features)
                    and the second tensor is the edge list of shape (num_edges, 2)
        Returns: a tensor of shape (num_edges, 2, num_features) where the first dimension is the edge index
                 and the second dimension is the node index
        """

        node_features, edge_list = inputs
        x = ops.take(node_features, edge_list, axis=0)
        return x


class ReduceGatheredNodesSum(layers.Layer):
    """
    Reduces the gathered nodes to a single tensor
    """

    def __init__(self, **kwargs):
        super(ReduceGatheredNodesSum, self).__init__(**kwargs)

    def compute_output_shape(self, inputs_shape):
        return inputs_shape[0]

    def call(self, inputs):
        """
        Computes the sum of the gathered nodes as new node features.

        For each edge of the form (i, j), the feature of the node j
        is added to the new representation of the node i.

        Args:
            inputs: a list of tensors:
                - the first tensor is the node features of shape (num_nodes, num_features)
                - the second tensor is the gathered nodes of shape (num_edges, 2, num_features)
                - the third tensor is the edge list of shape (num_edges, 2)

        Returns: a tensor of shape (num_nodes, num_features)
        """

        node_features, gathered_nodes, edge_list = inputs

        incoming_features = gathered_nodes[:, 0]

        outgoing_indices = edge_list[:, 1]
        outgoing_indices = ops.expand_dims(outgoing_indices, axis=-1)

        results_incoming = ops.scatter(
            outgoing_indices, incoming_features, node_features.shape
        )

        results = results_incoming

        return results


class ReduceNodeSum(layers.Layer):
    """
    Reduces the node features to a single graph feature
    """

    def __init__(self, **kwargs):
        super(ReduceNodeSum, self).__init__(**kwargs)

    def compute_output_shape(self, inputs_shape):
        # just remove the num_nodes dimension
        shape = list(inputs_shape)
        shape.pop(-2)

        return tuple(shape)

    def call(self, inputs):
        """
        Computes the sum of the node features

        Args:
            inputs: a tensor of shape (num_nodes, num_features)

        Returns: a tensor of shape (num_features,)
        """

        return ops.einsum("ijk->ik", inputs)


class ApplyOverBatch(layers.Layer):
    def __init__(self, layer, **kwargs):
        if "name" not in kwargs:
            name = f"{layer.name}_over_batch"
        else:
            name = kwargs.pop("name")
        super(ApplyOverBatch, self).__init__(name=name, **kwargs)
        self.layer = layer
        self.built = False

    def build(self, input_shape):
        if not self.built:
            super(ApplyOverBatch, self).build(input_shape)

            # also build the child layer
            child_input_shape = [shape[1:] for shape in input_shape]
            self.layer.build(child_input_shape)

            self.built = True

    def compute_output_shape(self, inputs_shape):
        # remove the batch size dimension
        unbacthed_shape = [shape[1:] for shape in inputs_shape]

        sublayer_output = self.layer.compute_output_shape(unbacthed_shape)

        # add the batch size dimension back
        return tuple([inputs_shape[0][0]] + list(sublayer_output))

    def call(self, inputs):
        """
        Applies the layer to each graph in the batch

        Args:
            inputs: a list of tensors. Each tensor has shape (batch_size, ...).
                    The tensors will be unstacked and the layer will be applied on zip(*unstacked_inputs)

        Returns: a tensor of shape (batch_size, ...). Returns whatever the child layer returns,
                but stacked back into a batched tensor.

        """

        return ops.stack(
            [
                self.layer(graph_data)
                for graph_data in zip(*[ops.unstack(tensor) for tensor in inputs])
            ]
        )

    def get_config(self):
        config = super(ApplyOverBatch, self).get_config()
        config.update({"layer": keras.saving.serialize_keras_object(self.layer)})

        return config

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("layer")
        sublayer = keras.saving.deserialize_keras_object(sublayer_config)
        instance = cls(layer=sublayer, **config)
        instance.built = True
        return instance

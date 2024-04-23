import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from model.layers.convolution import *
from model.layers.common_layers import *
from model.data.read_dataset import read_dataset
from keras import ops
from keras import optimizers

from collections import Counter


import keras
import numpy as np
import tensorflow as tf


# Load the dataset
dataset = "model/data/NCI109"

node_attributes_ds, edge_list_ds, degrees_ds, graph_features_ds = read_dataset(
    dataset, include_node_attributes=False, include_node_labels=True
)

print("Node attributes shape:", node_attributes_ds.shape)


# create a tf dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (
        {
            "adjacency_inputs": edge_list_ds,
            "node_inputs": node_attributes_ds,
            "degrees": degrees_ds,
        },
        graph_features_ds,
    )
)

# shuffle and batch the dataset
dataset = dataset.shuffle(buffer_size=4096, reshuffle_each_iteration=True).batch(
    32, drop_remainder=True
)

# print the number of batches
print("Number of batches:", len(list(dataset)))

# split into train and validation
dataset_val = dataset.take(20)
dataset = dataset.skip(20)

# print the number of batches
print("Number of batches:", len(list(dataset)))
print("Validation batches:", len(list(dataset_val)))

print("Degree shape:", degrees_ds.shape)

edge_list_inputs_og = keras.Input(
    shape=edge_list_ds[0].shape, name="adjacency_inputs", dtype="int32"
)
node_inputs_og = keras.Input(shape=node_attributes_ds[0].shape, name="node_inputs")
degrees_inputs_og = keras.Input(shape=degrees_ds[0].shape, name="degrees")

node_representation_layer = node_inputs_og

node_representation_layer = ApplyOverBatch(
    SingleGraphConvolution(256, activation=ops.relu, name="convolution")
)([node_representation_layer, edge_list_inputs_og, degrees_inputs_og])

residual = node_representation_layer

# a few convolutions
for i in range(1):
    node_representation_layer = ApplyOverBatch(
        SingleGraphConvolution(256, activation=ops.relu, name=f"convolution{i}")
    )([node_representation_layer, edge_list_inputs_og, degrees_inputs_og])


residual = keras.layers.Add()([node_representation_layer, residual])

# reduce the node features to a single graph feature
graph_features = ReduceNodeSum()(residual)
# graph_features = ExpandDims(axis=0)(graph_features)

# a few dense layers
graph_features = keras.layers.Dense(256, activation="relu", name="dense")(
    graph_features
)

graph_features = keras.layers.Dense(256, activation="relu", name="dense2")(
    graph_features
)


# the output layer
outputs = keras.layers.Dense(
    graph_features_ds.shape[-1], activation="softmax", name="output"
)(graph_features)


model = keras.Model(
    inputs=[edge_list_inputs_og, node_inputs_og, degrees_inputs_og],
    outputs=outputs,
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary(expand_nested=True)

# Train the model
model.fit(
    dataset,
    epochs=10,
    validation_data=dataset_val,
)

data_distribution = np.array(list(Counter(graph_features_ds.argmax(axis=1)).values()))

print("Data distribution: ", data_distribution)
print("Relative distribution: ", data_distribution / np.sum(data_distribution))

model.save("model.keras")

# Load the model
model = keras.models.load_model("model.keras")

model.summary()

# Evaluate the model
model.evaluate(dataset_val)

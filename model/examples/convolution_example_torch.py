import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

import os

os.environ["KERAS_BACKEND"] = "torch"

from model.layers.convolution import *
from model.layers.common_layers import *
from model.data.read_dataset import read_dataset
from keras import ops
from keras import optimizers

from collections import Counter

import keras
import numpy as np


# Load the dataset
dataset = "model/data/NCI109"

node_attributes_ds, edge_list_ds, degrees_ds, graph_features_ds = read_dataset(
    dataset, include_node_attributes=False, include_node_labels=True
)

# split the dataset into training and validation
split = 0.8
split_idx = int(len(node_attributes_ds) * split)

node_attributes_ds, node_attributes_val = (
    node_attributes_ds[:split_idx],
    node_attributes_ds[split_idx:],
)

edge_list_ds, edge_list_val = edge_list_ds[:split_idx], edge_list_ds[split_idx:]

degrees_ds, degrees_val = degrees_ds[:split_idx], degrees_ds[split_idx:]

graph_features_ds, graph_features_val = (
    graph_features_ds[:split_idx],
    graph_features_ds[split_idx:],
)

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
    [edge_list_ds, node_attributes_ds, degrees_ds],
    graph_features_ds,
    batch_size=32,
    epochs=10,
    validation_data=(
        [edge_list_val, node_attributes_val, degrees_val],
        graph_features_val,
    ),
)

data_distribution = np.array(list(Counter(graph_features_ds.argmax(axis=1)).values()))

print("Data distribution: ", data_distribution)
print("Relative distribution: ", data_distribution / np.sum(data_distribution))

model.save("model.keras")

# Load the model
model = keras.models.load_model("model.keras")

print(model.summary())

# Evaluate the model
model.evaluate(([edge_list_val, node_attributes_val, degrees_val], graph_features_val))

### \file
### \ingroup tutorial_tmva
### \notebook -nodraw
###
### Example of getting batches of events from a ROOT dataset into a basic
### TensorFlow workflow.
###
### \macro_code
### \macro_output
### \author Dante Niewenhuis

import tensorflow as tf
import ROOT

tree_name = "sig_tree"
file_name = "http://root.cern/files/Higgs_data.root"

batch_size = 128
chunk_size = 5_000

target = "Type"

# Returns two TF.Dataset for training and validation batches.
ds_train, ds_valid = ROOT.TMVA.Experimental.CreateTFDatasets(
    tree_name,
    file_name,
    batch_size,
    chunk_size,
    validation_split=0.3,
    target=target,
)

# Get a list of the columns used for training
input_columns = ds_train.train_columns
num_features = len(input_columns)

##############################################################################
# AI example
##############################################################################

# Define TensorFlow model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            300, activation=tf.nn.tanh, input_shape=(num_features,)
        ),  # input shape required
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)
loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

# Train model
model.fit(ds_train, validation_data=ds_valid, epochs=2)

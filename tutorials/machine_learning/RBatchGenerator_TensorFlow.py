### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### Example of getting batches of events from a ROOT dataset into a basic
### TensorFlow workflow.
###
### \macro_code
### \macro_output
### \author Dante Niewenhuis

import ROOT

# TensorFlow has to be imported after ROOT to avoid LLVM symbol clashes if ROOT
# was built with LLVM in Debug mode and TensorFlow>=2.20.0.
import tensorflow as tf

tree_name = "sig_tree"
file_name = str(ROOT.gROOT.GetTutorialDir()) + "/machine_learning/data/Higgs_data.root"

batch_size = 128
chunk_size = 5000
block_size = 300

rdataframe = ROOT.RDataFrame(tree_name, file_name)
target = ["Type"]

# Returns two TF.Dataset for training and validation batches.
ds_train, ds_valid = ROOT.TMVA.Experimental.CreateTFDatasets(
    rdataframe,
    batch_size,    
    chunk_size,
    block_size,        
    target = target,
    validation_split = 0.3,
    shuffle = True,
    drop_remainder = True
)

num_of_epochs = 2

# Datasets have to be repeated as many times as there are epochs
ds_train_repeated = ds_train.repeat(num_of_epochs)
ds_valid_repeated = ds_valid.repeat(num_of_epochs)

# Number of batches per epoch must be given for model.fit
train_batches_per_epoch = ds_train.number_of_batches
validation_batches_per_epoch = ds_valid.number_of_batches

# Get a list of the columns used for training
input_columns = ds_train.train_columns
num_features = len(input_columns)

##############################################################################
# AI example
##############################################################################

# Define TensorFlow model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(300, activation=tf.nn.tanh),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)

loss_fn = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(ds_train_repeated, steps_per_epoch=train_batches_per_epoch, validation_data=ds_valid_repeated,\
          validation_steps=validation_batches_per_epoch, epochs=num_of_epochs)


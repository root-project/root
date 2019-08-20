## \file
## \ingroup tutorial_tmva
## \notebook -nodraw
## This tutorial shows how to perform an efficient BDT inference with modern
## interfaces.
##
## \macro_code
## \macro_output
##
## \date August 2019
## \author Stefan Wunsch

import ROOT
import numpy as np

# Load a BDT model from remote
# Note that this model was trained with the tutorial tmva101_Training.py.
bdt = ROOT.TMVA.Experimental.RBDT("MyBDT", "http://root.cern/files/tmva101_model.root");

# The model can now be applied in different scenarios:
# 1) Event-by-event inference
# 2) Batch inference on data of multiple events
# 3) Inference as part of an RDataFrame graph

# 1) Event-by-event inference
# The event-by-event inference takes the values of the variables as a std::vector<float>.
# In Python, you can alternatively pass a numpy.ndarray, which is converted in the back
# via memory-adoption (without a copy) to the according C++ type.
prediction = bdt.Compute(np.array([0.5, 1.0, -0.2, 1.5], dtype="float32"))
print("Single-event inference: {}".format(prediction))

# 2) Batch inference on data of multiple events
# For batch inference, the data needs to be structured as a matrix. For this
# purpose, TMVA makes use of the RTensor class. In Python, you can simply
# pass again a numpy.ndarray.
x = np.array([[0.5, 1.0, -0.2, 1.5],
              [0.1, 0.2, -0.5, 0.9],
              [0.0, 1.2, 0.1, -0.2]], dtype="float32")
y = bdt.Compute(x)

print("RTensor input for inference on data of multiple events:\n{}".format(x))
print("Prediction performed on multiple events:\n{}".format(y))

# 3) Perform inference as part of an RDataFrame graph
# TODO

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
variables = ROOT.std.vector["string"](("var1", "var2", "var3", "var4"))

def helper(model, num_variables):
    # TODO: Move to the pythonization layer
    seq = ROOT.std.integer_sequence("std::size_t", *range(num_variables))()
    seqtype = type(seq).__cppname__
    modeltype = type(model).__cppname__
    return ROOT.TMVA.Experimental.Internal.ComputeHelper[seqtype, "float", modeltype + "&"](ROOT.std.move(model))

def make_histo(treename):
    df = ROOT.RDataFrame(treename, "http://root.cern.ch/files/tmva_class_example.root")
    df = df.Define("y", helper(bdt, len(variables)), variables)
    return df.Histo1D((treename, ";BDT score;N_{Events}", 30, -0.5, 0.5), "y")

sig = make_histo("TreeS")
bkg = make_histo("TreeB")

# Make plot
ROOT.gStyle.SetOptStat(0)
c = ROOT.TCanvas("", "", 800, 800)

sig.SetLineColor(ROOT.kRed)
bkg.SetLineColor(ROOT.kBlue)
sig.SetLineWidth(2)
bkg.SetLineWidth(2)
bkg.Draw("HIST")
sig.Draw("HIST SAME")

legend = ROOT.TLegend(0.7, 0.7, 0.89, 0.89)
legend.SetBorderSize(0)
legend.AddEntry("TreeS", "Signal", "l")
legend.AddEntry("TreeB", "Background", "l")
legend.Draw()

c.DrawClone()

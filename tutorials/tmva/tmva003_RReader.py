## \file
## \ingroup tutorial_tmva
## \notebook -nodraw
## This tutorial shows how to apply with the modern interfaces models saved in
## TMVA XML files.
##
## \macro_code
## \macro_output
##
## \date April 2022
## \author Harshal Shende

import ROOT
import numpy as np
from ROOT import TMVA


TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()


def train(filename):
    # Create factory
    output = ROOT.TFile.Open("TMVA.root", "RECREATE")
    factory = ROOT.TMVA.Factory("tmva003", output, V=False, DrawProgressBar=False, AnalysisType="Classification")

    # Open trees with signal and background events
    data = ROOT.TFile.Open(filename)
    signal = data.Get("TreeS")
    background = data.Get("TreeB")

    # Add variables and register the trees with the dataloader
    dataloader = ROOT.TMVA.DataLoader("tmva003_BDT")
    variables = ["var1", "var2", "var3", "var4"]

    for var in variables:
        dataloader.AddVariable(var)

    dataloader.AddSignalTree(signal, 1.0)
    dataloader.AddBackgroundTree(background, 1.0)
    dataloader.PrepareTrainingAndTestTree("", "")

    # Train a TMVA method
    factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, "BDT", V=False, H=False, NTrees=300, MaxDepth=2)
    factory.TrainAllMethods()


# First, let's train a model with TMVA.
filename = "http://root.cern.ch/files/tmva_class_example.root"
train(filename)

# Next, we load the model from the TMVA XML file.
model = ROOT.TMVA.Experimental.RReader("tmva003_BDT/weights/tmva003_BDT.weights.xml")

# In case you need a reminder of the names and order of the variables during training, you can ask the model for it.
variables = model.GetVariableNames()

# The model can now be applied in different scenarios:
# 1) Event-by-event inference
# 2) Batch inference on data of multiple events
# 3) Inference as part of an RDataFrame graph

# 1) Event-by-event inference
# The event-by-event inference takes the values of the variables as a std::vector<float>.
# Note that the return value is as well a std::vector<float> since the reader
# is also capable to process models with multiple outputs.
prediction = model.Compute({0.5, 1.0, -0.2, 1.5})
print("Single-event inference: {}".format(prediction[0]))

# 2) Batch inference on data of multiple events
# For batch inference, the data needs to be structured as a matrix. For this
# purpose, TMVA makes use of the RTensor class. For convenience, we use RDataFrame
# and the AsTensor utility to make the read-out from the ROOT file.
df = ROOT.RDataFrame("TreeS", filename).AsNumpy()
df = np.vstack([df[var] for var in variables]).T  # Read only a small subset of the dataset
x = ROOT.std.vector["std::vector<float>"]()
vector = np.vectorize(np.float)
for i in range(3):
    x.push_back(vector(df[i]))
y = model.Compute(x)
print("RTensor input for inference on data of multiple events:\n", x)
print("Prediction performed on multiple events: ", y)


# 3) Perform inference as part of an RDataFrame graph
# We write a small lambda function that performs for us the inference on
# a dataframe to omit code duplication.
def make_histo(treename):
    df2 = df.Define("y", ROOT.TMVA.Experimental.Compute(model), variables)
    return df2.Histo1D((treename.c_str(), ";BDT score;N_{Events}", 30, -0.5, 0.5), "y")


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

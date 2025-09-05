## \file
## \ingroup tutorial_ml
## \notebook -nodraw
## This tutorial illustrates how you can test a trained BDT model using the fast
## tree inference engine offered by TMVA and external tools such as scikit-learn.
##
## \macro_code
## \macro_output
##
## \date August 2019
## \author Stefan Wunsch

import ROOT
import pickle

from tmva100_DataPreparation import variables
from tmva101_Training import load_data


# Load data
x, y_true, w = load_data("test_signal.root", "test_background.root")

# Load trained model
File = "tmva101.root"

bdt = ROOT.TMVA.Experimental.RBDT("myBDT", File)

# Make prediction
y_pred = bdt.Compute(x)

# Compute ROC using sklearn
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_pred, sample_weight=w)
score = auc(false_positive_rate, true_positive_rate)

# Plot ROC
c = ROOT.TCanvas("roc", "", 600, 600)
g = ROOT.TGraph(len(false_positive_rate), false_positive_rate, true_positive_rate)
g.SetTitle("AUC = {:.2f}".format(score))
g.SetLineWidth(3)
g.SetLineColor("kRed")
g.Draw("AC")
g.GetXaxis().SetRangeUser(0, 1)
g.GetYaxis().SetRangeUser(0, 1)
g.GetXaxis().SetTitle("False-positive rate")
g.GetYaxis().SetTitle("True-positive rate")
c.Draw()

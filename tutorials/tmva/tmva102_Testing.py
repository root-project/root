## \file
## \ingroup tutorial_tmva
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
bdt = ROOT.TMVA.Experimental.RBDT[""]("myBDT", "tmva101.root")

# Make prediction
y_pred = bdt.Compute(x)

# Compute ROC using sklearn
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_true, y_pred, sample_weight=w)
score = auc(fpr, tpr, reorder=True)

# Plot ROC
c = ROOT.TCanvas("roc", "", 600, 600)
g = ROOT.TGraph(len(fpr), fpr, tpr)
g.SetTitle("AUC = {:.2f}".format(score))
g.SetLineWidth(3)
g.SetLineColor(ROOT.kRed)
g.Draw("AC")
g.GetXaxis().SetRangeUser(0, 1)
g.GetYaxis().SetRangeUser(0, 1)
g.GetXaxis().SetTitle("False-positive rate")
g.GetYaxis().SetTitle("True-positive rate")
c.Draw()

## \file
## \ingroup tutorial_dataframe
## \notebook -draw
## This tutorial shows how VecOps can be used to slim down the programming
## model typically adopted in HEP for analysis.
##
## \macro_code
## \macro_image
##
## \date March 2018
## \author Danilo Piparo, Andre Vieira Silva

import ROOT
from math import sqrt

filename = ROOT.gROOT.GetTutorialDir().Data() + "/dataframe/df017_vecOpsHEP.root"
treename = "myDataset"
RDF = ROOT.ROOT.RDataFrame

def WithPyROOT():
    f = ROOT.TFile(filename)
    h = ROOT.TH1F("pt", "pt", 16, 0, 4)
    for event in f.myDataset:
        for E, px, py in zip(event.E, event.px, event.py):
            if (E > 100):
               h.Fill(sqrt(px*px + py*py))
    h.DrawCopy()

def WithRDataFrameVecOpsJit():
    f = RDF(treename, filename)
    h = f.Define("good_pt", "sqrt(px*px + py*py)[E>100]")\
         .Histo1D(("pt", "pt", 16, 0, 4), "good_pt")
    h.DrawCopy()

## We plot twice the same quantity, the key is to look into the implementation
## of the functions above
c = ROOT.TCanvas()
c.Divide(2,1)
c.cd(1)
WithPyROOT()
c.cd(2)
WithRDataFrameVecOpsJit()

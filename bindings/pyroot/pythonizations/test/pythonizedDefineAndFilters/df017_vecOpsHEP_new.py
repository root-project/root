import ROOT
import numpy as np


filename = ROOT.gROOT.GetTutorialDir().Data() + "/dataframe/df017_vecOpsHEP.root"
treename = "myDataset"

# # @ROOT.Numba.Declare(["RVecD","RVecD","RVecD"],"RVecD")
# def good_pt(px, py, E)->"RVecD":
#     # px = np.array([float(x) for x in px])
#     # py = np.array([float(x) for x in py])
#     # E = np.array([float(x) for x in E])
#     p = px*px + py*py
#     return p[E>100]

def WithPyROOT(filename):
    from math import sqrt
    f = ROOT.TFile(filename)
    h = ROOT.TH1F("pt", "With PyROOT", 16, 0, 4)
    
    for event in f.myDataset:
        for E, px, py in zip(event.E, event.px, event.py):
            if (E > 100):
               h.Fill(sqrt(px*px + py*py))
    h.DrawCopy()

def WithRDataFrameVecOpsJit(treename, filename):
    f = ROOT.RDataFrame(treename, filename)
    def good_pt(px, py, E, Elim)->"RVecD":
        p = px*px + py*py
        return p[E>Elim]
    f = f.Define("good_pt",good_pt, {'Elim':100})
    h =  f.Histo1D(("pt", "With RDataFrame and RVec", 16, 0, 4), "good_pt")
    h.DrawCopy()

## We plot twice the same quantity, the key is to look into the implementation
## of the functions above
c = ROOT.TCanvas()
c.Divide(2,1)
c.cd(1)
WithPyROOT(filename)
c.cd(2)
WithRDataFrameVecOpsJit(treename, filename)
c.SaveAs("df017_vecOpsHEP_new.png")
"""rootlogon module for the Python tutorials. Disables the graphics."""
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gROOT.SetWebDisplay("batch")

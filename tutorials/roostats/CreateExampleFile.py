import ROOT

ROOT.gROOT.ProcessLine(".! prepareHistFactory .")
ROOT.gROOT.ProcessLine(".! hist2workspace config/example.xml")

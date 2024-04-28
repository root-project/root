import ROOT
gROOT = ROOT.gROOT

def CreateExampleFile():
   print("preparing HistFactory ...")
   gROOT.ProcessLine(".! prepareHistFactory .")
   print("... done.""")
   print("moving HistoryFactory to Working Space ...")
   gROOT.ProcessLine(".! hist2workspace config/example.xml")
   print("... done.")


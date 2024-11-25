import ROOT

gROOT = ROOT.gROOT


def CreateExampleFile():

    print("Preparing HistFactory ...")
    gROOT.ProcessLine(".! prepareHistFactory .")
    print("... done." "")

    print("Moving HistoryFactory to Working Space ...")
    gROOT.ProcessLine(".! hist2workspace config/example.xml")
    print("... done.")

# this is meant to be used only to run tutorials as tests

import ROOT
import sys


nCores = int(sys.argv[1])
tutorialName = sys.argv[2]

if "imt" in ROOT.gROOT.GetConfigFeatures():
    ROOT.EnableImplicitMT(nCores)

tutorialFile = open(tutorialName)
exec(tutorialFile.read())

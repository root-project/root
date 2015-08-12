#!/usr/bin/python

"""Command line to open a ROOT file on a TBrowser"""

from cmdLineUtils import *

# Help strings
COMMAND_HELP = \
    "Open a ROOT file on a TBrowser " + \
    "(for more informations please look at the man page)."

##### Beginning of the main code #####

# Collect arguments with the module argparse
parser = argparse.ArgumentParser(description=COMMAND_HELP)
parser.add_argument("sourceName", nargs='?', help=SOURCE_HELP)
args = parser.parse_args()

if args.sourceName:
    with stderrRedirected():
        rootFile = ROOT.TFile.Open(args.sourceName)
    rootFile.Browse(ROOT.TBrowser())
    ROOT.PyROOT.TPyROOTApplication.Run(ROOT.gApplication)
    rootFile.Close()
else :
    browser = ROOT.TBrowser()
    ROOT.PyROOT.TPyROOTApplication.Run(ROOT.gApplication)

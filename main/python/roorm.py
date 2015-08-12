#!/usr/bin/python

"""Command line to remove
objects from ROOT files"""

from cmdLineUtils import *

# Help strings
COMMAND_HELP = \
    "Remove objects from ROOT files " + \
    "(for more informations please look at the man page)."
I_HELP = \
    "prompt before every removal."

##### Beginning of the main code #####

# Collect arguments with the module argparse
parser = argparse.ArgumentParser(description=COMMAND_HELP)
parser.add_argument("sourcePatternList", help=SOURCES_HELP, nargs='+')
parser.add_argument("-i", help=I_HELP, action="store_true")
args = parser.parse_args()

# Create a list of tuples that contain source ROOT file names
# and lists of path in these files
sourceList = \
    [tup for pattern in args.sourcePatternList \
    for tup in patternToFileNameAndPathSplitList(pattern)]

# Create a dictionnary with options
optDict = vars(args)

# Loop on the ROOT files
for fileName, pathSplitList in sourceList:
    with stderrRedirected():
        rootFile = ROOT.TFile.Open(fileName,"update")
    for pathSplit in pathSplitList:
        deleteRootObject(rootFile,pathSplit,optDict)
    rootFile.Close()

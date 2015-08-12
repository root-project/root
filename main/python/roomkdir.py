#!/usr/bin/python

"""Command line to add directories in ROOT files"""

from cmdLineUtils import *

def createDirectory(rootFile,pathSplit):
    """Add a directory named 'pathSplit[-1]'
    in (rootFile,pathSplit[:-1])"""
    changeDirectory(rootFile,pathSplit[:-1])
    ROOT.gDirectory.mkdir(pathSplit[-1])

def createDirectories(rootFile,pathSplit,optDict):
    """Same behaviour as createDirectory but allows the possibility
    to build an whole path recursively with opt_dict["parents"]"""
    if not optDict["parents"] and pathSplit != []:
        createDirectory(rootFile,pathSplit)
    else:
        for i in range(len(pathSplit)):
            currentPathSplit = pathSplit[:i+1]
            objNameList = \
                [key.GetName() for key in \
                getKeyList(rootFile,currentPathSplit[:-1])]
            if not currentPathSplit[-1] in objNameList:
                createDirectory(rootFile,currentPathSplit)

# Help strings
COMMAND_HELP = \
    "Add directories in a ROOT files " + \
    "(for more informations please look at the man page)."
PARENT_HELP = \
    "make parent directories as needed, no error if existing."

##### Beginning of the main code #####

# Collect arguments with the module argparse
parser = argparse.ArgumentParser(description=COMMAND_HELP)
parser.add_argument("sourcePatternList", help=SOURCES_HELP, nargs='+')
parser.add_argument("-p", "--parents", help=PARENT_HELP, action="store_true")
args = parser.parse_args()

# Create a list of tuples that contain source ROOT file names
# and lists of path in these files
sourceList = \
    [tup for pattern in args.sourcePatternList \
    for tup in patternToFileNameAndPathSplitList(pattern,wildcards=False)]

# Create a dictionnary with options
optDict = vars(args)

# Loop on the ROOT files
for fileName, pathSplitList in sourceList:
    with stderrRedirected():
        rootFile = ROOT.TFile.Open(fileName,"update")
    for pathSplit in pathSplitList:
        createDirectories(rootFile,pathSplit,optDict)
    rootFile.Close()

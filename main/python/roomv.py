#!/usr/bin/python

"""Command line to move objects
from ROOT files to an other"""

from cmdLineUtils import *

# Help strings

COMMAND_HELP = \
    "Move objects from ROOT files to an other " + \
    "(for more informations please look at the man page)."

##### Beginning of the main code #####

# Collect arguments with the module argparse
parser = argparse.ArgumentParser(description=COMMAND_HELP)
parser.add_argument("sourcePatternList", help=SOURCES_HELP, nargs='+')
parser.add_argument("destPattern", help=DEST_HELP)
parser.add_argument("-c","--compress", type=int, help=COMPRESS_HELP)
parser.add_argument("--replace", help="", action="store_true")
parser.add_argument("--recreate", help=RECREATE_HELP, action="store_true")
args = parser.parse_args()

# Create a list of tuples that contain source ROOT file names
# and lists of path in these files
sourceList = \
    [tup for pattern in args.sourcePatternList \
    for tup in patternToFileNameAndPathSplitList(pattern)]

# Create a tuple that contain a destination ROOT file name
# and a path in this file
destList = \
    patternToFileNameAndPathSplitList( \
    args.destPattern,wildcards=False)
destFileName,destPathSplitList = destList[0]
destPathSplit = destPathSplitList[0]

# Create a dictionnary with options
optDict = vars(args)

# Change the compression settings only on non existing file
if optDict["compress"] and os.path.isfile(destFileName):
    logging.error("can't change compression settings on existing file")
    sys.exit()

# Creation of destination file (changing of the compression settings)
with stderrRedirected(): destFile = \
    ROOT.TFile.Open(destFileName,"recreate") \
    if optDict["recreate"] else \
    ROOT.TFile.Open(destFileName,"update")
if optDict["compress"]: destFile.SetCompressionSettings(optDict["compress"])
ROOT.gROOT.GetListOfFiles().Remove(destFile) # Fast copy necessity


# Loop on the root files
for sourceFileName, sourcePathSplitList in sourceList:
    with stderrRedirected(): sourceFile = \
        ROOT.TFile.Open(sourceFileName,"update") \
        if sourceFileName != destFileName else \
        destFile
    ROOT.gROOT.GetListOfFiles().Remove(sourceFile) # Fast copy necessity
    for sourcePathSplit in sourcePathSplitList:
        oneSource = len(sourceList)==1 and len(sourcePathSplitList)==1
        copyRootObject(sourceFile,sourcePathSplit, \
            destFile,destPathSplit,optDict,oneSource)
        deleteRootObject(sourceFile,sourcePathSplit)
    if sourceFileName != destFileName:
        sourceFile.Close()
destFile.Close()

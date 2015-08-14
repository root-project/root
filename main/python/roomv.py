#!/usr/bin/env python

# ROOT command line tools module: cmdLineUtils
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to move objects from ROOT files to another"""

import os
import cmdLineUtils

# Help strings

COMMAND_HELP = "Move objects from ROOT files to another"

EPILOG = """Examples:
- roomv source.root:hist* dest.root
  Move all histograms whose named starts with 'hist' from 'source.root' to 'dest.root'.

- roomv source1.root:hist1 source2.root:hist2 dest.root
  Move histograms 'hist1' from 'source1.root' and 'hist2' from 'source2.root' to 'dest.root'.

- roomv --recreate source.root:hist dest.root
  Recreate the destination file 'dest.root' and move the histogram named 'hist' from 'source.root' into it.

- roomv -c 1 source.root:hist dest.root
  Change the compression level of the destination file 'dest.root' and move the histogram named 'hist' from 'source.root' into it. For more information about compression settings of ROOT file, please look at the reference guide available on the ROOT site.
"""

REPLACE_HELP = "Replace existing object"
INTERACTIVE_HELP = "prompt before overwrite."
RECURSIVE_HELP = "move directories recursively"

##### Beginning of the main code #####

# Collect arguments with the module argparse
parser = argparse.ArgumentParser(description=COMMAND_HELP)
parser.add_argument("sourcePatternList", help=SOURCES_HELP, nargs='+')
parser.add_argument("destPattern", help=DEST_HELP)
parser.add_argument("-c","--compress", type=int, help=COMPRESS_HELP)
parser.add_argument("-i","--interactive", help=INTERACTIVE_HELP, action="store_true")
parser.add_argument("--recreate", help=RECREATE_HELP, action="store_true")
parser.add_argument("-r","--recursive", help=RECURSIVE_HELP, action="store_true")
parser.add_argument("--replace", help="", action="store_true")
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
compressOptionValue = optDict["compress"]

# Change the compression settings only on non existing file
if compressOptionValue != None and os.path.isfile(destFileName):
    logging.error("can't change compression settings on existing file")
    sys.exit()

# Creation of destination file (changing of the compression settings)
with stderrRedirected(): destFile = \
    ROOT.TFile(destFileName,"recreate") \
    if optDict["recreate"] else \
    ROOT.TFile(destFileName,"update")
if compressOptionValue != None: destFile.SetCompressionSettings(compressOptionValue)
ROOT.gROOT.GetListOfFiles().Remove(destFile) # Fast copy necessity


# Loop on the root files
for sourceFileName, sourcePathSplitList in sourceList:
    with stderrRedirected(): sourceFile = \
        ROOT.TFile(sourceFileName,"update") \
        if sourceFileName != destFileName else \
        destFile
    ROOT.gROOT.GetListOfFiles().Remove(sourceFile) # Fast copy necessity
    for sourcePathSplit in sourcePathSplitList:
        oneSource = len(sourceList)==1 and len(sourcePathSplitList)==1
        copyRootObject(sourceFile,sourcePathSplit, \
            destFile,destPathSplit,optDict,oneSource)
        deleteRootObject(sourceFile,sourcePathSplit,optDict)
    if sourceFileName != destFileName:
        sourceFile.Close()
destFile.Close()

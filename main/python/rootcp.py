#!/usr/bin/env python

# ROOT command line tools: rootcp
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to copy objects from ROOT files into an other"""

import sys
import os
import logging
import ROOT
import cmdLineUtils

# Help strings
COMMAND_HELP = "Copy objects from ROOT files into an other"

EPILOG = """Examples:
- roocp source.root:hist* dest.root
  Copy all histograms whose named starts with 'hist' from 'source.root' to 'dest.root'.

- roocp source1.root:hist1 source2.root:hist2 dest.root
  Copy histograms 'hist1' from 'source1.root' and 'hist2' from 'source2.root' to 'dest.root'.

- roocp --recreate source.root:hist dest.root
  Recreate the destination file 'dest.root' and copy the histogram named 'hist' from 'source.root' into it.

- roocp -c 1 source.root:hist dest.root
  Change the compression factor of the destination file 'dest.root' if not existing and copy the histogram named 'hist' from 'source.root' into it.
"""

def copyObjects(fileName, pathSplitList, destFile, destPathSplit, optDict, oneFile):
    retcode = 0
    destFileName = destFile.GetName()
    rootFile = cmdLineUtils.openROOTFile(fileName) \
        if fileName != destFileName else \
        destFile
    if not rootFile: return 1
    ROOT.gROOT.GetListOfFiles().Remove(rootFile) # Fast copy necessity
    for pathSplit in pathSplitList:
        oneSource = oneFile and len(pathSplitList)==1
        retcode += cmdLineUtils.copyRootObject(rootFile,pathSplit, \
            destFile,destPathSplit,optDict,oneSource)
    if fileName != destFileName: rootFile.Close()
    return retcode

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserSourceDest(COMMAND_HELP, EPILOG)
    parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
    parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
    parser.add_argument("-r","--recursive", help=cmdLineUtils.RECURSIVE_HELP, action="store_true")
    parser.add_argument("--replace", help=cmdLineUtils.REPLACE_HELP, action="store_true")

    # Put arguments in shape
    sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)
    if sourceList == [] or destFileName == "": return 1
    if optDict["recreate"] and destFileName in [n[0] for n in sourceList]:
        logging.error("cannot recreate destination file if this is also a source file")
        return 1

    # Open destination file
    destFile = cmdLineUtils.openROOTFileCompress(destFileName,optDict)
    if not destFile: return 1
    ROOT.gROOT.GetListOfFiles().Remove(destFile) # Fast copy necessity

    # Loop on the root files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += copyObjects(fileName, pathSplitList, destFile, \
                               destPathSplit, optDict, len(sourceList)==1)
    destFile.Close()
    return retcode

sys.exit(execute())

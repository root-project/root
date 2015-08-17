#!/usr/bin/env python

# ROOT command line tools: roomv
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to move objects from ROOT files to another"""

import sys
import os
import logging
import ROOT
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

MOVE_ERROR = "error during copy of {0}, it is not removed from {1}"

def moveObjects(fileName, pathSplitList, destFile, destPathSplit, optDict, oneFile):
    retcode = 0
    destFileName = destFile.GetName()
    rootFile = cmdLineUtils.openROOTFile(fileName,"update") \
        if fileName != destFileName else \
        destFile
    if not rootFile: return 1
    ROOT.gROOT.GetListOfFiles().Remove(rootFile) # Fast copy necessity
    for pathSplit in pathSplitList:
        oneSource = oneFile and len(pathSplitList)==1
        retcodeTemp = cmdLineUtils.copyRootObject(rootFile,pathSplit, \
            destFile,destPathSplit,optDict,oneSource)
        if not retcodeTemp:
            retcode += cmdLineUtils.deleteRootObject(rootFile,pathSplit,optDict)
        else:
            logging.warning(MOVE_ERROR.format("/".join(pathSplit),rootFile.GetName()))
            retcode += retcodeTemp
    if fileName != destFileName: rootFile.Close()
    return retcode

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserSourceDest(COMMAND_HELP, EPILOG)
    parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
    parser.add_argument("-i","--interactive", help=cmdLineUtils.INTERACTIVE_HELP, action="store_true")
    parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
    
    # Put arguments in shape
    sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)
    if sourceList == [] or destFileName == "": return 1
    if optDict["recreate"] and destFileName in sourceList:
        logging.error("cannot recreate destination file if this is also a source file")
        return 1
    optDict["recursive"] = True
    optDict["replace"] = True

    # Open destination file
    destFile = cmdLineUtils.openROOTFileCompress(destFileName,optDict)    
    if not destFile: return 1
    ROOT.gROOT.GetListOfFiles().Remove(destFile) # Fast copy necessity

    # Loop on the root files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += moveObjects(fileName, pathSplitList, destFile, \
                               destPathSplit, optDict, len(sourceList)==1)
    destFile.Close()
    return retcode

sys.exit(execute())

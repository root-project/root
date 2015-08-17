#!/usr/bin/env python

# ROOT command line tools: rooeventselector
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to copy subsets of trees from source
ROOT files to new trees on a destination ROOT file"""

import sys
import cmdLineUtils

# Help strings
COMMAND_HELP = "Copy subsets of trees from source ROOT files"

FIRST_EVENT_HELP = "specify the first event to copy"
LAST_EVENT_HELP = "specify the last event to copy"

EPILOG="""Examples:
- rooeventselector source.root:tree dest.root
  Copy the tree 'tree' from 'source.root' to 'dest.root'.

- rooeventselector -f 101 source.root:tree dest.root
  Copy a subset of the tree 'tree' from 'source.root' to 'dest.root'. The new tree contains events from the old tree except the first hundred.

- rooeventselector -l 100 source.root:tree dest.root
  Copy a subset of the tree  'tree' from 'source.root' to 'dest.root'. The new tree contains the first hundred events from the old tree.

- rooeventselector --recreate source.root:tree dest.root
  Recreate the destination file 'dest.root' and copy the tree 'tree' from 'source.root' to 'dest.root'.

- rooeventselector -c 1 source.root:tree dest.root
  Change the compression factor of the destination file 'dest.root' and  copy the tree 'tree' from 'source.root' to 'dest.root'. For more information about compression settings of ROOT file, please look at the reference guide available on the ROOT site.
"""

def copyTreeSubset(sourceFile,sourcePathSplit,destFile,destPathSplit,firstEvent,lastEvent):
    """Copy a subset of the tree from (sourceFile,sourcePathSplit)
    to (destFile,destPathSplit) according to options in optDict"""
    retcode = cmdLineUtils.changeDirectory(sourceFile,sourcePathSplit[:-1])
    if retcode != 0: return retcode
    bigTree = cmdLineUtils.getFromDirectory(sourcePathSplit[-1])
    nbrEntries = bigTree.GetEntries()
    # changeDirectory for the small tree not to be memory-resident
    retcode = cmdLineUtils.changeDirectory(destFile,destPathSplit)
    if retcode != 0: return retcode
    smallTree = bigTree.CloneTree(0)
    if lastEvent == -1:
        lastEvent = nbrEntries-1
    for i in xrange(firstEvent, lastEvent+1):
        bigTree.GetEntry(i)
        smallTree.Fill()
    smallTree.Write()
    return retcode
    
def copyTreeSubsets(fileName, pathSplitList, destFile, destPathSplit, optDict):
    retcode = 0
    destFileName = destFile.GetName()
    rootFile = cmdLineUtils.openROOTFile(fileName) \
        if fileName != destFileName else \
        destFile
    if not rootFile: return 1
    for pathSplit in pathSplitList:
        if cmdLineUtils.isTree(rootFile,pathSplit):
            retcode += copyTreeSubset(rootFile,pathSplit, \
            destFile,destPathSplit,optDict["first"],optDict["last"])
    if fileName != destFileName: rootFile.Close()
    return retcode

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserSourceDest(COMMAND_HELP, EPILOG)
    parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
    parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
    parser.add_argument("-f","--first", type=int, default=0,help=FIRST_EVENT_HELP)
    parser.add_argument("-l","--last", type=int, default=-1, help=LAST_EVENT_HELP)
    
    # Put arguments in shape
    sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)
    if sourceList == [] or destFileName == "": return 1
    if optDict["recreate"] and destFileName in sourceList:
        logging.error("cannot recreate destination file if this is also a source file")
        return 1
        
    # Open destination file
    destFile = cmdLineUtils.openROOTFileCompress(destFileName,optDict)    
    if not destFile: return 1

    # Loop on the root file
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += copyTreeSubsets(fileName, pathSplitList, destFile, destPathSplit, optDict)
    destFile.Close()
    return retcode

sys.exit(execute())

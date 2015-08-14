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

FIRST_EVENT_HELP = "specify the first event to copy"
LAST_EVENT_HELP = "specify the last event to copy"


def copyTreeSubset(sourceFile,sourcePathSplit,destFile,destPathSplit,firstEvent,lastEvent):
    """Copy a subset of the tree from (sourceFile,sourcePathSplit)
    to (destFile,destPathSplit) according to options in optDict"""
    cmdLineUtils.changeDirectory(sourceFile,sourcePathSplit[:-1])
    bigTree = cmdLineUtils.getFromDirectory(sourcePathSplit[-1])
    nbrEntries = bigTree.GetEntries()
    # changeDirectory for the small tree not to be memory-resident
    cmdLineUtils.changeDirectory(destFile,destPathSplit)
    smallTree = bigTree.CloneTree(0)
    if lastEvent == -1:
        lastEvent = nbrEntries-1

    for i in xrange(firstEvent, lastEvent+1):
        bigTree.GetEntry(i)
        smallTree.Fill()

    smallTree.Write()

def execute():
    parser = cmdLineUtils.getParserSourceDest(COMMAND_HELP, EPILOG)
    parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
    parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
    parser.add_argument("-f","--first", type=int, default=0,help=FIRST_EVENT_HELP)
    parser.add_argument("-l","--last", type=int, default=-1, help=LAST_EVENT_HELP)

    sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)
    compressOptionValue = optDict["compress"]

    retcode = 0

    # Change the compression settings only on non existing file
    if compressOptionValue != None and os.path.isfile(destFileName):
        logging.error("can't change compression settings on existing file")
        return 1

    # Creation of destination file (changing of the compression settings)
    mode = "recreate" if optDict["recreate"] else "update"

    with cmdLineUtils.stderrRedirected():
        destFile = cmdLineUtils.openROOTFile(destFileName,mode)

    if not destfile: return 1

    if compressOptionValue != None: destFile.SetCompressionSettings(compressOptionValue)

    # Loop on the root file
    for sourceFileName, sourcePathSplitList in sourceList:
        with cmdLineUtils.stderrRedirected(): sourceFile = \
            cmdLineUtils.openROOTFile(sourceFileName) \
            if sourceFileName != destFileName else \
            destFile

        if not sourceFile:
            retcode += 1
            continue

        for sourcePathSplit in sourcePathSplitList:
            if cmdLineUtils.isTree(sourceFile,sourcePathSplit): copyTreeSubset( \
                sourceFile,sourcePathSplit, \
                destFile,destPathSplit,optDict["first"],optDict["last"])
        if sourceFileName != destFileName:
            sourceFile.Close()
    destFile.Close()

    return retcode

sys.exit(execute())

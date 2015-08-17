#!/usr/bin/env python

# ROOT command line tools: roorm
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to remove objects from ROOT files"""

import sys
import cmdLineUtils

# Help strings
COMMAND_HELP = "Remove objects from ROOT files"

EPILOG = """Examples:
- roorm example.root:hist
  Remove the object 'hist' from the ROOT file 'example.root'

- roorm example.root:dir/hist
  Remove the object 'hist' from the direcory 'dir' inside the ROOT file 'example.root'

- roorm example.root
  Remove the ROOT file 'example.root'

- roorm -i example.root:hist
  Display a confirmation request before deleting: 'remove 'hist' from 'example.root' ? (y/n) :'
"""

def removeObjects(fileName, pathSplitList, optDict):
    retcode = 0
    rootFile = cmdLineUtils.openROOTFile(fileName,"update")
    if not rootFile: return 1
    for pathSplit in pathSplitList:
        retcode += cmdLineUtils.deleteRootObject(rootFile,pathSplit,optDict)
    rootFile.Close()
    return retcode

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserFile(COMMAND_HELP, EPILOG)
    parser.add_argument("-i","--interactive", help=cmdLineUtils.INTERACTIVE_HELP, action="store_true")
    parser.add_argument("-r","--recursive", help=cmdLineUtils.RECURSIVE_HELP, action="store_true")

    # Put arguments in shape
    sourceList, optDict = cmdLineUtils.getSourceListOptDict(parser)
    if sourceList == []: return 1

    # Loop on the root files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += removeObjects(fileName, pathSplitList,optDict)
    return retcode

sys.exit(execute())

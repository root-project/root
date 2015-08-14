#!/usr/bin/env python

"""Command line to add directories in ROOT files"""

import sys
import ROOT
import cmdLineUtils

# Help strings
COMMAND_HELP = "Add directories in ROOT files"

EPILOG="""Examples:
- roomkdir example.root:dir
  Add the directory 'dir' to the ROOT file 'example.root'

- roomkdir example.root:dir1/dir2
  Add the directory 'dir2' in 'dir1' which is into the ROOT file 'example.root'

- roomkdir -p example.root:dir1/dir2/dir3
  Make parent directories of 'dir3' as needed, no error if existing

- roomkdir example.root
  Create an empty ROOT file named 'example.root'
"""

PARENT_HELP = "make parent directories as needed, no error if existing."

def createDirectories(rootFile,pathSplit,optDict):
    """Same behaviour as createDirectory but allows the possibility
    to build an whole path recursively with opt_dict["parents"]"""
    retcode = 0
    if not optDict["parents"] and pathSplit != []:
        retcode += cmdLineUtils.createDirectory(rootFile,pathSplit)
    else:
        for i in range(len(pathSplit)):
            currentPathSplit = pathSplit[:i+1]
            objNameList = \
                [key.GetName() for key in \
                cmdLineUtils.getKeyList(rootFile,currentPathSplit[:-1])]
            if not currentPathSplit[-1] in objNameList:
                retcode += cmdLineUtils.createDirectory(rootFile,currentPathSplit)
    return retcode

def processFile(fileName, pathSplitList,optDict):
    retcode = 0
    with cmdLineUtils.stderrRedirected():
        rootFile = cmdLineUtils.openROOTFile(fileName,"update")
        if not rootFile: return 1
    for pathSplit in pathSplitList:
        retcode+=createDirectories(rootFile,pathSplit,optDict)
    rootFile.Close()
    return retcode

def execute():
    parser = cmdLineUtils.getParserFile(COMMAND_HELP, EPILOG)
    parser.add_argument("-p", "--parents", help=PARENT_HELP, action="store_true")

    sourceList, optDict = cmdLineUtils.getSourceListOptDict(parser, wildcards = False)

    # Loop on the ROOT files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += processFile(fileName, pathSplitList,optDict)
    return retcode

sys.exit(execute())

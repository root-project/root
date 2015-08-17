#!/usr/bin/env python

# ROOT command line tools: roomkdir
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to add directories in ROOT files"""

import sys
import logging
import cmdLineUtils

# Help strings
COMMAND_HELP = "Add directories in ROOT files"

PARENT_HELP = "make parent directories as needed, no error if existing."

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

MKDIR_ERROR = "cannot create directory '{0}'"

def createDirectories(rootFile,pathSplit,optDict):
    """Same behaviour as createDirectory but allows the possibility
    to build an whole path recursively with opt_dict["parents"]"""
    retcode = 0
    lenPathSplit = len(pathSplit)
    if lenPathSplit == 0:
        pass
    elif optDict["parents"]:
        for i in xrange(lenPathSplit):
            currentPathSplit = pathSplit[:i+1]
            if not (cmdLineUtils.isExisting(rootFile,currentPathSplit) \
                and cmdLineUtils.isDirectory(rootFile,currentPathSplit)):
                retcode += cmdLineUtils.createDirectory(rootFile,currentPathSplit)
    else:
        doMkdir = True
        for i in xrange(lenPathSplit-1):
            currentPathSplit = pathSplit[:i+1]
            if not (cmdLineUtils.isExisting(rootFile,currentPathSplit) \
                and cmdLineUtils.isDirectory(rootFile,currentPathSplit)):
                doMkdir = False
                break
        if doMkdir:
            retcode += cmdLineUtils.createDirectory(rootFile,pathSplit)
        else:
            logging.warning(MKDIR_ERROR.format("/".join(pathSplit)))
            retcode += 1
    return retcode

def processFile(fileName, pathSplitList, optDict):
    retcode = 0
    rootFile = cmdLineUtils.openROOTFile(fileName,"update")
    if not rootFile: return 1
    for pathSplit in pathSplitList:
        retcode+=createDirectories(rootFile,pathSplit,optDict)
    rootFile.Close()
    return retcode

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserFile(COMMAND_HELP, EPILOG)
    parser.add_argument("-p", "--parents", help=PARENT_HELP, action="store_true")

    # Put arguments in shape
    sourceList, optDict = cmdLineUtils.getSourceListOptDict(parser, wildcards = False)
    if sourceList == []: return 1

    # Loop on the ROOT files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += processFile(fileName, pathSplitList,optDict)
    return retcode

sys.exit(execute())

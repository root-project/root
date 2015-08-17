#!/usr/bin/env python

# ROOT command line tools: roobrowse
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to open a ROOT file on a TBrowser"""

import sys
import cmdLineUtils

# Help strings
COMMAND_HELP = "Open a ROOT file in a TBrowser"

EPILOG = """Examples:
- roobrowse
  Open a TBrowser
  
- roobrowse file.root
  Open the ROOT file 'file.root' in a TBrowser
"""

def openBrowser(rootFile=None):
    browser = cmdLineUtils.ROOT.TBrowser()
    if rootFile: rootFile.Browse(browser)
    cmdLineUtils.ROOT.PyROOT.TPyROOTApplication.Run(cmdLineUtils.ROOT.gApplication)

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserSingleFile(COMMAND_HELP, EPILOG)

    # Put arguments in shape
    args = cmdLineUtils.getArgs(parser)

    if args.FILE:
        rootFile = cmdLineUtils.openROOTFile(args.FILE)
        if not rootFile:
            return 1
        openBrowser(rootFile)
        rootFile.Close()
    else :
        openBrowser()
    return 0

sys.exit(execute())

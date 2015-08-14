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

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserFile(COMMAND_HELP)
    sourceList, args = cmdLineUtils.getSourceListArgs(parser)

    if args.FILE:
        with cmdLineUtils.stderrRedirected():
            rootFile = cmdLineUtils.openROOTFile(args.FILE)
        if not rootfile:
            return 1
        cmdLineUtils.openBrowser(rootFile)
        rootFile.Close()
    else :
        cmdLineUtils.openBrowser()
    return 0

sys.exit(execute())

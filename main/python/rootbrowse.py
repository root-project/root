#!/usr/bin/env python

# ROOT command line tools: rootbrowse
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to open a ROOT file on a TBrowser"""

import cmdLineUtils
import sys

# Help strings
COMMAND_HELP = "Open a ROOT file in a TBrowser"

EPILOG = """Examples:
- rootbrowse
  Open a TBrowser

- rootbrowse file.root
  Open the ROOT file 'file.root' in a TBrowser
"""

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserSingleFile(COMMAND_HELP, EPILOG)

    # Put arguments in shape
    args = cmdLineUtils.getArgs(parser)

    # Process rootBrowse
    return cmdLineUtils.rootBrowse(args.FILE)

sys.exit(execute())

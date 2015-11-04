#!/usr/bin/env python

# ROOT command line tools: rootrm
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to remove objects from ROOT files"""

import cmdLineUtils
import sys

# Help strings
COMMAND_HELP = "Remove objects from ROOT files"

EPILOG = """Examples:
- rootrm example.root:hist
  Remove the object 'hist' from the ROOT file 'example.root'

- rootrm example.root:dir/hist
  Remove the object 'hist' from the direcory 'dir' inside the ROOT file 'example.root'

- rootrm example.root
  Remove the ROOT file 'example.root'

- rootrm -i example.root:hist
  Display a confirmation request before deleting: 'remove 'hist' from 'example.root' ? (y/n) :'
"""

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserFile(COMMAND_HELP, EPILOG)
    parser.add_argument("-i","--interactive", help=cmdLineUtils.INTERACTIVE_HELP, action="store_true")
    parser.add_argument("-r","--recursive", help=cmdLineUtils.RECURSIVE_HELP, action="store_true")

    # Put arguments in shape
    sourceList, optDict = cmdLineUtils.getSourceListOptDict(parser)

    # Process rootRm
    return cmdLineUtils.rootRm(sourceList, interactive=optDict["interactive"], \
                               recursive=optDict["recursive"])

sys.exit(execute())

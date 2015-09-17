#!/usr/bin/env python

# ROOT command line tools: rootls
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to dump ROOT files contents to terminal"""

import cmdLineUtils
import sys

# Help strings
COMMAND_HELP = """Display ROOT files contents in the terminal."""

ONE_HELP = "Print content in one column"
LONG_PRINT_HELP = "use a long listing format."
TREE_PRINT_HELP = "print tree recursively and use a long listing format."

EPILOG = """Examples:
- rootls example.root
  Display contents of the ROOT file 'example.root'.

- rootls example.root:dir
  Display contents of the directory 'dir' from the ROOT file 'example.root'.

- rootls example.root:*
  Display contents of the ROOT file 'example.root' and his subdirectories.

- rootls file1.root file2.root
  Display contents of ROOT files 'file1.root' and 'file2.root'.

- rootls *.root
  Display contents of ROOT files whose name ends with '.root'.

- rootls -1 example.root
  Display contents of the ROOT file 'example.root' in one column.

- rootls -l example.root
  Display contents of the ROOT file 'example.root' and use a long listing format.

- rootls -t example.root
  Display contents of the ROOT file 'example.root', use a long listing format and print trees recursively.
"""

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserFile(COMMAND_HELP, EPILOG)
    parser.add_argument("-1", "--oneColumn", help=ONE_HELP, action="store_true")
    parser.add_argument("-l", "--longListing", help=LONG_PRINT_HELP, action="store_true")
    parser.add_argument("-t", "--treeListing", help=TREE_PRINT_HELP, action="store_true")

    # Put arguments in shape
    sourceList, optDict = cmdLineUtils.getSourceListOptDict(parser)

    # Process rootLs
    return cmdLineUtils.rootLs(sourceList, oneColumn=optDict["oneColumn"], \
                               longListing=optDict["longListing"], treeListing=optDict["treeListing"])

sys.exit(execute())

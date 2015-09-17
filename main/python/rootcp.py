#!/usr/bin/env python

# ROOT command line tools: rootcp
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to copy objects from ROOT files into an other"""

import cmdLineUtils
import sys

# Help strings
COMMAND_HELP = "Copy objects from ROOT files into an other"

EPILOG = """Examples:
- roocp source.root:hist* dest.root
  Copy all histograms whose named starts with 'hist' from 'source.root' to 'dest.root'.

- roocp source1.root:hist1 source2.root:hist2 dest.root
  Copy histograms 'hist1' from 'source1.root' and 'hist2' from 'source2.root' to 'dest.root'.

- roocp --recreate source.root:hist dest.root
  Recreate the destination file 'dest.root' and copy the histogram named 'hist' from 'source.root' into it.

- roocp -c 1 source.root:hist dest.root
  Change the compression factor of the destination file 'dest.root' if not existing and copy the histogram named 'hist' from 'source.root' into it.
"""

def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserSourceDest(COMMAND_HELP, EPILOG)
    parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
    parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
    parser.add_argument("-r","--recursive", help=cmdLineUtils.RECURSIVE_HELP, action="store_true")
    parser.add_argument("--replace", help=cmdLineUtils.REPLACE_HELP, action="store_true")

    # Put arguments in shape
    sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)

    # Process rootCp
    return cmdLineUtils.rootCp(sourceList, destFileName, destPathSplit, \
                               compress=optDict["compress"], recreate=optDict["recreate"], \
                               recursive=optDict["recursive"], replace=optDict["replace"])

sys.exit(execute())

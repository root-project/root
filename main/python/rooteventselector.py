#!/usr/bin/env @python@

# ROOT command line tools: rooteventselector
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

# Additions
# Author: Lawrence Lee
# Mail: lawrence.lee.jr@cern.ch
# Date: 1/4/16

"""Command line to copy subsets of trees from source ROOT files to new trees on a destination ROOT file"""

import cmdLineUtils
import sys


# Help strings
description = "Copy subsets of trees from source ROOT files"

FIRST_EVENT_HELP = "specify the first event to copy"
LAST_EVENT_HELP = "specify the last event to copy"

EPILOG="""Examples:
- rooteventselector source.root:tree dest.root
  Copy the tree 'tree' from 'source.root' to 'dest.root'.

- rooteventselector -f 101 source.root:tree dest.root
  Copy a subset of the tree 'tree' from 'source.root' to 'dest.root'. The new tree contains events from the old tree except the first hundred.

- rooteventselector -l 100 source.root:tree dest.root
  Copy a subset of the tree  'tree' from 'source.root' to 'dest.root'. The new tree contains the first hundred events from the old tree.

- rooteventselector --recreate source.root:tree dest.root
  Recreate the destination file 'dest.root' and copy the tree 'tree' from 'source.root' to 'dest.root'.

- rooteventselector -c 1 source.root:tree dest.root
  Change the compression factor of the destination file 'dest.root' and  copy the tree 'tree' from 'source.root' to 'dest.root'. For more information about compression settings of ROOT file, please look at the reference guide available on the ROOT site.

- rooteventselector -s "(branch1Value > 100)&&( branch2Value )" source.root:tree dest.root
  Copy the tree 'tree' from 'source.root' to 'dest.root' and apply a selection to the output tree.

- rooteventselector -e "muon_*" source.root:tree dest.root
  Copy the tree 'tree' from 'source.root' to 'dest.root' and remove branches matching "muon_*"

- rooteventselector -e "*" -i "muon_*" source.root:tree dest.root
  Copy the tree 'tree' from 'source.root' to 'dest.root' and only write branches matching "muon_*"
"""

def get_argparse():
	# Collect arguments with the module argparse
	parser = cmdLineUtils.getParserSourceDest(description, EPILOG)
	parser.prog = 'rooteventselector'
	parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
	parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
	parser.add_argument("-f","--first", type=int, default=0, help=FIRST_EVENT_HELP)
	parser.add_argument("-l","--last", type=int, default=-1, help=LAST_EVENT_HELP)
	parser.add_argument("-s","--selection", default="")
	parser.add_argument("-i","--branchinclude", default="")
	parser.add_argument("-e","--branchexclude", default="")
	return parser

def execute():
	parser = get_argparse()
	# Put arguments in shape
	sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)

	# Process rootEventselector
	return cmdLineUtils.rootEventselector(sourceList, destFileName, destPathSplit, \
										compress=optDict["compress"], recreate=optDict["recreate"], \
										first=optDict["first"], last=optDict["last"], \
										selectionString=optDict["selection"], \
										branchinclude=optDict["branchinclude"],\
										branchexclude=optDict["branchexclude"])
if __name__ == "__main__":
	sys.exit(execute())

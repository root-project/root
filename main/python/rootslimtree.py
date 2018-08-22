#!/usr/bin/env @python@

# ROOT command line tools: rootslimtree
# Author: Lawrence Lee via Julien Ripoche's rooteventselector
# Mail: lawrence.lee.jr@cern.ch
# Date: 4/4/16

"""Command line to copy trees with subset of branches from source ROOT files to new trees on a destination ROOT file"""

import cmdLineUtils
import sys

# Help strings
COMMAND_HELP = "Copy trees with a subset of branches from source ROOT files"

EPILOG="""Examples:
- rootslimtree source.root:tree dest.root
  Copy the tree 'tree' from 'source.root' to 'dest.root'.

- rootslimtree --recreate source.root:tree dest.root
  Recreate the destination file 'dest.root' and copy the tree 'tree' from 'source.root' to 'dest.root'.

- rootslimtree -c 1 source.root:tree dest.root
  Change the compression factor of the destination file 'dest.root' and  copy the tree 'tree' from 'source.root' to 'dest.root'. For more information about compression settings of ROOT file, please look at the reference guide available on the ROOT site.

- rootslimtree -e "muon_*" source.root:tree dest.root
  Copy the tree 'tree' from 'source.root' to 'dest.root' and remove branches matching "muon_*"

- rootslimtree -e "*" -i "muon_*" source.root:tree dest.root
  Copy the tree 'tree' from 'source.root' to 'dest.root' and only write branches matching "muon_*"
"""

def get_argparse():
	# Collect arguments with the module argparse
	parser = cmdLineUtils.getParserSourceDest(COMMAND_HELP, EPILOG)
	parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
	parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
	parser.add_argument("-i","--branchinclude", default="")
	parser.add_argument("-e","--branchexclude", default="")
	
	return parser
	
def execute():
	parser = get_argparse()
	
	# Put arguments in shape
	sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)

	# Process rootEventselector in simplified slimtree mode
	return cmdLineUtils.rootEventselector(sourceList, destFileName, destPathSplit, \
										compress=optDict["compress"], recreate=optDict["recreate"], \
										first=0, last=-1, \
										selectionString="", \
										branchinclude=optDict["branchinclude"],\
										branchexclude=optDict["branchexclude"])
if __name__ == "__main__":
	sys.exit(execute())

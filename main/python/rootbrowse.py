#!/usr/bin/env @python@

# ROOT command line tools: rootbrowse
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to open a ROOT file on a TBrowser"""

import cmdLineUtils
import sys


# Help strings
description = "Open a ROOT file in a TBrowser"

EPILOG = """Examples:
- rootbrowse
  Open a TBrowser

- rootbrowse file.root
  Open the ROOT file 'file.root' in a TBrowser
"""

def get_argparse():
	# Collect arguments with the module argparse
	parser = cmdLineUtils.getParserSingleFile(description, EPILOG)
	parser.prog = 'rootbrowse'
	return parser


def execute():
	parser = get_argparse()

	# Put arguments in shape
	args = cmdLineUtils.getArgs(parser)

	# Process rootBrowse
	return cmdLineUtils.rootBrowse(args.FILE)

if __name__ == "__main__":
	sys.exit(execute())

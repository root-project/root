#!/usr/bin/env @python@

# ROOT command line tools: rootbrowse
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to open a ROOT file on a TBrowser"""

import cmdLineUtils
import sys

import ROOT

# Help strings
description = "Open a ROOT file in a TBrowser"

WEBON_HELP = "Configure webdisplay like chrome or qt6web"

WEBOFF_HELP = "Invoke the normal TBrowser (not the web version)"

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

	parser.add_argument("-w", "--web", help=WEBON_HELP)
	parser.add_argument("-wf", "--webOff", help=WEBOFF_HELP, action="store_true")
	return parser


def execute():
	parser = get_argparse()

	# Put arguments in shape
	args = cmdLineUtils.getArgs(parser)
	if args.webOff:
		ROOT.gROOT.SetWebDisplay("off")
	elif args.web:
		ROOT.gROOT.SetWebDisplay(args.web)


	# Process rootBrowse
	return cmdLineUtils.rootBrowse(args.FILE)

if __name__ == "__main__":
	sys.exit(execute())

#!/usr/bin/env @python@

# ROOT command line tools: rootprint
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to print ROOT files contents on ps,pdf or png,gif..."""

import cmdLineUtils
import sys

# Help strings
description = "Print ROOT files contents on ps,pdf or pictures files"

DIRECTORY_HELP = "put output files in a subdirectory named DIRECTORY."
DIVIDE_HELP = "divide the canvas ont the format 'x','y' (ex: 2,2)"
DRAW_HELP = "specify draw option"
FORMAT_HELP = "specify output format (ex: pdf, png)."
OUTPUT_HELP = "merge files in a file named OUTPUT (only for ps and pdf)."
SIZE_HELP = "specify canvas size on the format 'width'x'height' (ex: 600x400)"
STYLE_HELP = "specify a C file name which define a style"
VERBOSE_HELP = "print informations about the running"

EPILOG = """Examples:
- rootprint example.root:hist
  Create a pdf file named 'hist.pdf' which contain the histogram 'hist'.

- rootprint -d histograms example.root:hist
  Create a pdf file named 'hist.pdf' which contain the histogram 'hist' and put it in the directory 'histograms' (create it if not already exists).

- rootprint -f png example.root:hist
  Create a png file named 'hist.png' which contain the histogram 'hist'.

- rootprint -o histograms.pdf example.root:hist*
  Create a pdf file named 'histograms.pdf' which contain all histograms whose name starts with 'hist'. It works also with postscript.
"""

def get_argparse():
	# Collect arguments with the module argparse
	parser = cmdLineUtils.getParserFile(description, EPILOG)
	parser.prog = 'rootprint'
	parser.add_argument("-d", "--directory", help=DIRECTORY_HELP)
	parser.add_argument("--divide", help=DIVIDE_HELP)
	parser.add_argument("-D", "--draw", default="",  help=DRAW_HELP)
	parser.add_argument("-f", "--format", help=FORMAT_HELP)
	parser.add_argument("-o", "--output", help=OUTPUT_HELP)
	parser.add_argument("-s", "--size", help=SIZE_HELP)
	parser.add_argument("-S", "--style", help=STYLE_HELP)
	parser.add_argument("-v", "--verbose", action="store_true", help=VERBOSE_HELP)
	return parser

def execute():
	parser = get_argparse()

	# Put arguments in shape
	sourceList, optDict = cmdLineUtils.getSourceListOptDict(parser)

	# Process rootPrint
	return cmdLineUtils.rootPrint(sourceList, directoryOption = optDict["directory"], \
								divideOption = optDict["divide"], drawOption = optDict["draw"], \
								formatOption = optDict["format"], \
								outputOption = optDict["output"], sizeOption = optDict["size"], \
								styleOption = optDict["style"], verboseOption = optDict["verbose"])
if __name__ == "__main__":
	sys.exit(execute())

#!/usr/bin/env @python@

# ROOT command line tools: rootmv
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to move objects from ROOT files to another"""

import cmdLineUtils
import sys

# Help strings
description = "Move objects from ROOT files to another"

EPILOG = """Examples:
- rootmv source.root:hist* dest.root
  Move all histograms whose named starts with 'hist' from 'source.root' to 'dest.root'.

- rootmv source1.root:hist1 source2.root:hist2 dest.root
  Move histograms 'hist1' from 'source1.root' and 'hist2' from 'source2.root' to 'dest.root'.

- rootmv --recreate source.root:hist dest.root
  Recreate the destination file 'dest.root' and move the histogram named 'hist' from 'source.root' into it.

- rootmv -c 101 source.root:hist dest.root
  Change the compression settings of the destination file 'dest.root' to ZLIB algorithm with compression level 1 and move the histogram named 'hist' from 'source.root' into it.
  Meaning of the '-c' argument is given by 'compress = 100 * algorithm + level'.
  Other examples of usage:
    * -c 509 : ZSTD with compression level 9
    * -c 404 : LZ4 with compression level 4
    * -c 207 : LZMA with compression level 7
  For more information see https://root.cern.ch/doc/master/classTFile.html#ad0377adf2f3d88da1a1f77256a140d60
  and https://root.cern.ch/doc/master/structROOT_1_1RCompressionSetting.html

  """

def get_argparse():
	# Collect arguments with the module argparse
	parser = cmdLineUtils.getParserSourceDest(description, EPILOG)
	parser.prog = 'rootmv'
	parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
	parser.add_argument("-i","--interactive", help=cmdLineUtils.INTERACTIVE_HELP, action="store_true")
	parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
	return parser

def execute():
	parser = get_argparse()

	# Put arguments in shape
	sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)

	# Process rootMv
	return cmdLineUtils.rootMv(sourceList, destFileName, destPathSplit, \
								compress=optDict["compress"], interactive=optDict["interactive"], \
								recreate=optDict["recreate"])
if __name__ == "__main__":
	sys.exit(execute())

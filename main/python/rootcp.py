#!/usr/bin/env @python@

# ROOT command line tools: rootcp
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Command line to copy objects from ROOT files into an other"""

import cmdLineUtils
import sys


# Help strings
description = "Copy objects from ROOT files into an other"

EPILOG = """
Note: If an object has been written to a file multiple times, rootcp will copy only the latest version of that object.

Examples:
- rootcp source.root dest.root
  Copy the latest version of each object in 'source.root' to 'dest.root'.

- rootcp source.root:hist* dest.root
  Copy all histograms whose names start with 'hist' from 'source.root' to 'dest.root'.

- rootcp source1.root:hist1 source2.root:hist2 dest.root
  Copy histograms 'hist1' from 'source1.root' and 'hist2' from 'source2.root' to 'dest.root'.

- rootcp --recreate source.root:hist dest.root
  Recreate 'dest.root' and copy the histogram named 'hist' from 'source.root' into it.

- rootcp -c 101 source.root:hist dest.root
  Change compression, if not existing, of 'dest.root' to ZLIB algorithm with compression level 1 and copy the histogram named 'hist' from 'source.root' into it.
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
	parser.prog = 'rootcp'
	parser.add_argument("-c","--compress", type=int, help=cmdLineUtils.COMPRESS_HELP)
	parser.add_argument("--recreate", help=cmdLineUtils.RECREATE_HELP, action="store_true")
	parser.add_argument("-r","--recursive", help=cmdLineUtils.RECURSIVE_HELP, action="store_true")
	parser.add_argument("--replace", help=cmdLineUtils.REPLACE_HELP, action="store_true")
	return parser


def execute():
	parser = get_argparse()
	 # Put arguments in shape
	sourceList, destFileName, destPathSplit, optDict = cmdLineUtils.getSourceDestListOptDict(parser)

	# Process rootCp
	return cmdLineUtils.rootCp(sourceList, destFileName, destPathSplit, \
				compress=optDict["compress"], recreate=optDict["recreate"], \
				recursive=optDict["recursive"], replace=optDict["replace"])
if __name__ == "__main__":
	sys.exit(execute())

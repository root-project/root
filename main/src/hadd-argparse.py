import argparse
import textwrap

def get_argparse():
	DESCRIPTION = """This program will add histograms from a list of root files and write them to a target root file.\n
The target file is newly created and must not exist, or if -f (\"force\") is given, must not be one of the source files.\n
"""
	EPILOGUE = """
If Target and source files have different compression settings a slower method is used.
For options that takes a size as argument, a decimal number of bytes is expected.
If the number ends with a ``k'', ``m'', ``g'', etc., the number is multiplied by 1000 (1K), 1000000 (1MB), 1000000000 (1G), etc.
If this prefix is followed by i, the number is multiplied by the traditional 1024 (1KiB), 1048576 (1MiB), 1073741824 (1GiB), etc.
The prefix can be optionally followed by B whose casing is ignored, eg. 1k, 1K, 1Kb and 1KB are the same.
"""
	parser = argparse.ArgumentParser(add_help=False, prog='hadd',
	description = DESCRIPTION, epilog = EPILOGUE)
	parser.add_argument("-a", help="Append to the output")
	parser.add_argument("-k", help="Skip corrupt or non-existent files, do not exit")
	parser.add_argument("-T", help="Do not merge Trees")
	parser.add_argument("-O", help="Re-optimize basket size when merging TTree")
	parser.add_argument("-v", help=textwrap.fill(
		"Explicitly set the verbosity level: 0 request no output, 99 is the default"))
	parser.add_argument("-j", help="Parallelize the execution in multiple processes")
	parser.add_argument("-dbg", help=textwrap.fill(
		"Parallelize the execution in multiple processes in debug mode "
		"(Does not delete partial files stored inside working directory)"))
	parser.add_argument("-d", help=textwrap.fill(
		"Carry out the partial multiprocess execution in the specified directory"))
	parser.add_argument("-n", help=textwrap.fill(
		"Open at most 'maxopenedfiles' at once (use 0 to request to use the system maximum)"))
	parser.add_argument("-cachesize", help=textwrap.fill(
		"Resize the prefetching cache use to speed up I/O operations(use 0 to disable)"))
	parser.add_argument("-experimental-io-features", help=textwrap.fill(
		"Used with an argument provided, enables the corresponding experimental feature for output trees"))
	parser.add_argument("-f", help=textwrap.fill(
		"Gives the ability to specify the compression level of the target file (by default 4) "))
	parser.add_argument("-fk", help=textwrap.fill(
		"Sets the target file to contain the baskets with the same compression "
		"as the input files (unless -O is specified). Compresses the meta data "
		"using the compression level specified in the first input or the "
		"compression setting after fk (for example 206 when using -fk206)"))
	parser.add_argument("-ff", help="The compression level use is the one specified in the first input")
	parser.add_argument("-f0", help="Do not compress the target file")
	parser.add_argument("-f6", help=textwrap.fill(
		"Use compression level 6. (See TFile::SetCompressionSettings for the support range of value.)"))
	parser.add_argument("TARGET", help="Target file")
	parser.add_argument("SOURCES", help="Source files")
	return parser

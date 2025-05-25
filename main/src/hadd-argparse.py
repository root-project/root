import argparse
import textwrap


def get_argparse():
    DESCRIPTION = textwrap.fill(
        "This program will add histograms, trees and other objects from a list "
        "of ROOT files and write them to a target ROOT file. The target file is "
        "newly created and must not exist, or if -f (\"force\") is given, must "
        "not be one of the source files.")

    EPILOGUE = textwrap.fill(
        "If TARGET and SOURCES have different compression settings a slower "
        "method is used. For options that takes a size as argument, a decimal "
        "number of bytes is expected. If the number ends with a ``k'', ``m'', "
        "``g'', etc., the number is multiplied by 1000 (1K), 1000000 (1MB), "
        "1000000000 (1G), etc. If this prefix is followed by i, the number is "
        "multiplied by the traditional 1024 (1KiB), 1048576 (1MiB), 1073741824 "
        "(1GiB), etc. The prefix can be optionally followed by B whose casing "
        "is ignored, eg. 1k, 1K, 1Kb and 1KB are the same. ")
    parser = argparse.ArgumentParser(add_help=False, prog='hadd',
                                     description=DESCRIPTION, epilog=EPILOGUE)
    parser.add_argument("-a", help="Append to the output", action = 'store_true')
    parser.add_argument("-f", help=textwrap.fill(
        "Force overwriting of output file"), action = 'store_true')
    parser.add_argument("-f[0-9]", help=textwrap.fill(
        "Gives the ability to specify the compression level of the target file. "
        "Default is 1 (kDefaultZLIB), 0 is uncompressed, 9 is maximum compression (see TFile::TFile documentation). "
        "You can also specify the full compresion algorithm, e.g. -f206"), action = 'store_true')
    parser.add_argument("-fk", help=textwrap.fill(
        "Sets the target file to contain the baskets with the same compression "
        "as the input files (unless -O is specified). Compresses the meta data "
        "using the compression level specified in the first input or the "
        "compression setting after fk (for example 206 when using -fk206)"), action = 'store_true')
    parser.add_argument("-ff", help="The compression level use is the one specified in the first input", action = 'store_true')
    parser.add_argument("-k", help="Skip corrupt or non-existent files, do not exit", action = 'store_true')
    parser.add_argument("-L", help=textwrap.fill(
       "Read the list of objects from FILE and either only merge or skip those objects depending on "
       "the value of \"-Ltype\". FILE must contain one object name per line, which cannot contain "
       "whitespaces or '/'. You can also pass TDirectory names, which apply to the entire directory "
       "content. Lines beginning with '#' are ignored. If this flag is passed, \"-Ltype\" MUST be "
       "passed as well."), action = 'store_true')
    parser.add_argument("-Ltype", help=textwrap.fill(
        "Sets the type of operation performed on the objects listed in FILE given with the "
        "\"-L\" flag. \"SkipListed\" will skip all the listed objects; \"OnlyListed\" will only merge those "
        "objects. If this flag is passed, \"-L\" must be passed as well."), action = 'store_true')
    parser.add_argument("-O", help="Re-optimize basket size when merging TTree", action = 'store_true')
    parser.add_argument("-T", help="Do not merge Trees", action = 'store_true')
    parser.add_argument("-v", help=textwrap.fill(
        "Explicitly set the verbosity level: 0 request no output, 99 is the default"))
    parser.add_argument("-j", help=textwrap.fill(
        "Parallelize the execution in 'J' processes. If the number of "
        "processes is not specified, use the system maximum."))
    parser.add_argument("-dbg", help=textwrap.fill(
        "Enable verbosity. If -j was specified, do not not delete partial files "
        "stored inside working directory."), action = 'store_true')
    parser.add_argument("-d", help=textwrap.fill(
        "Carry out the partial multiprocess execution in the specified directory"))
    parser.add_argument("-n", help=textwrap.fill(
        "Open at most 'N' files at once (use 0 to request to use the system maximum)"))
    parser.add_argument("-cachesize", help=textwrap.fill(
        "Resize the prefetching cache use to speed up I/O operations (use 0 to disable)"))
    parser.add_argument("-experimental-io-features", help=textwrap.fill(
        "Used with an argument provided, enables the corresponding experimental "
        "feature for output trees. See ROOT::Experimental::EIOFeatures"))
    parser.add_argument("TARGET", help="Target file")
    parser.add_argument("SOURCES", help="Source files")
    return parser

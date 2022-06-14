import argparse


def get_argparse():
    DESCRIPTION = "hist2workspace is a utility to create RooFit/RooStats workspace from histograms"
    parser = argparse.ArgumentParser(prog="hist2workspace", description=DESCRIPTION)
    parser.add_argument("-v", help="switch HistFactory message stream to INFO level", action="store_true")
    parser.add_argument("-vv", help="switch HistFactory message stream to DEBUG level", action="store_true")
    parser.add_argument(
        "-disable_binned_fit_optimization",
        help="disable the binned fit optimization used in HistFactory since ROOT 6.28",
        action="store_true",
    )
    return parser

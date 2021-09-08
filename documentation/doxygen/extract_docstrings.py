#-------------------------------------------------------------------------------
#  Author: Enric Tejedor <enric.tejedor.saavedra@cern.ch> CERN
#-------------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

# Code that extracts the docstrings from pythonization files and stores them
# in .pyzdoc files, so that doxygen can process them later to merge the
# documentation they contain with that of C++ files

import ast
import sys
from os import path

# Inspired from https://pypi.org/project/scandir/:
# Use the built-in version of scandir/walk if possible, otherwise
# use the scandir module
try:
    from os import scandir
except ImportError:
    from scandir import scandir

if len(sys.argv) < 3:
    print("Please provide the directory where documented .py files are, and the output folder for the .pyzdoc files.")
    exit(1)

pyz_dir = sys.argv[1]
pyz_dir_out = sys.argv[2]

def run_fast_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)


    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


_, filenames = run_fast_scandir(pyz_dir, [".py"])


# Iterate over pythonization files
for pyz_file_path in filenames:

    with open(pyz_file_path) as fd:
        file_contents = fd.read()

    # Docs for pythonizations are provided as a module-level docstring
    module = ast.parse(file_contents)
    ds = ast.get_docstring(module)
    if ds is not None:
        pyz_filename_out = pyz_dir_out + '/' + pyz_file_path[pyz_file_path.rfind("_"):] + 'zdoc'
        with open(pyz_filename_out, 'w') as pyz_doc_file:
            pyz_doc_file.write(ds)


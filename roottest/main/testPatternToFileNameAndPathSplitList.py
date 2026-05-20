#!/usr/bin/python

"""Test module for pattern to tuple"""

from cmdLineUtils import *

with stderrRedirected():
    from cmdLineUtils import patternToFileNameAndPathSplitList
    import argparse

# Collect arguments with the module argparse
parser = argparse.ArgumentParser(description="test for pattern to tuple function")
parser.add_argument("patternList", \
                    help="file path and object path in the file with the syntax : [filePath/]file[.root]:[objectPath/]object", \
                    nargs='+')
args = parser.parse_args()

# Create a list of tuples that contain a ROOT file name and a list of path in this file
fileList = []
for pattern in args.patternList:
    fileList.extend(patternToFileNameAndPathSplitList(pattern))

print(fileList)

#!/usr/bin/python

# ROOT command line tools: rooprint
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Command line to print ROOT files contents on ps,pdf or png,gif..."""

import sys
import ROOT
import logging
import os
import cmdLineUtils

# Help strings
COMMAND_HELP = "Print ROOT files contents on ps,pdf or pictures files"

DIRECTORY_HELP = "put output files in a subdirectory named DIRECTORY."
DRAW_HELP = "specify draw option"
FORMAT_HELP = "specify output format (ex: pdf, png)."
OUTPUT_HELP = "merge files in a file named OUTPUT (only for ps and pdf)."
SIZE_HELP = "specify canvas size on the format 'width'x'height' (ex: 600x400)"
VERBOSE_HELP = "print informations about the running"

EPILOG = """Examples:
- rooprint example.root:hist
  Create a pdf file named 'hist.pdf' which contain the histogram 'hist'.

- rooprint -d histograms example.root:hist
  Create a pdf file named 'hist.pdf' which contain the histogram 'hist' and put it in the directory 'histograms' (create it if not already exists).

- rooprint -f png example.root:hist
  Create a png file named 'hist.png' which contain the histogram 'hist'.

- rooprint -o histograms.pdf example.root:hist*
  Create a pdf file named 'histograms.pdf' which contain all histograms whose name starts with 'hist'. It works also with postscript.
"""

def keyListExtended(rootFile,pathSplitList):
    keyList,dirList = cmdLineUtils.keyClassSpliter(rootFile,pathSplitList)
    for pathSplit in dirList: keyList.extend(cmdLineUtils.getKeyList(rootFile,pathSplit))
    keyList = [key for key in keyList if not cmdLineUtils.isDirectoryKey(key)]
    cmdLineUtils.keyListSort(keyList)
    return keyList
    
def execute():
    # Collect arguments with the module argparse
    parser = cmdLineUtils.getParserFile(COMMAND_HELP, EPILOG)
    parser.add_argument("-d", "--directory", help=DIRECTORY_HELP)
    parser.add_argument("-D", "--draw", default="",  help=DRAW_HELP)
    parser.add_argument("-f", "--format", help=FORMAT_HELP)
    parser.add_argument("-o", "--output", help=OUTPUT_HELP)
    parser.add_argument("-s", "--size", help=SIZE_HELP)
    parser.add_argument("-v", "--verbose", action="store_true", help=VERBOSE_HELP)

    # Put arguments in shape
    sourceList, optDict = cmdLineUtils.getSourceListOptDict(parser)
    if sourceList == []: return 1
    cmdLineUtils.tupleListSort(sourceList)
    
    # Option values
    directoryOptionValue = optDict["directory"]
    drawOptionValue = optDict["draw"]
    formatOptionValue = optDict["format"]
    outputOptionValue = optDict["output"]
    sizeOptionValue = optDict["size"]

    # Verbose option
    if not optDict["verbose"]:
        ROOT.gErrorIgnoreLevel = 9999

    # Don't open windows
    ROOT.gROOT.SetBatch()

    # Initialize the canvas
    if sizeOptionValue:
        try:
            width,height = sizeOptionValue.split("x")
            width = int(width)
            height = int(height)
        except ValueError:
            logging.warning("canvas size is on a wrong format")
            return 1
        canvas = ROOT.TCanvas("canvas","canvas",width,height)
    else:
        canvas = ROOT.TCanvas("canvas")

    # Take the format of the output file (format option)
    if not formatOptionValue and outputOptionValue:
        fileName = outputOptionValue
        fileFormat = fileName.split(".")[-1]
        formatOptionValue = fileFormat

    # Use pdf as default format
    if not formatOptionValue: formatOptionValue = "pdf"

    # Create the output directory (directory option)
    if directoryOptionValue:
        if not os.path.isdir(os.path.join(os.getcwd(),directoryOptionValue)):
            os.mkdir(directoryOptionValue)

    # Make the output name, begin to print (output option)
    if outputOptionValue:
        if formatOptionValue in ['ps','pdf']:
            outputFileName = outputOptionValue
            if directoryOptionValue: outputFileName = \
                directoryOptionValue + "/" + outputFileName
            canvas.Print(outputFileName+"[",formatOptionValue)
        else:
            logging.warning("can't merge pictures, only postscript or pdf files")
            outputOptionValue = None

    # Loop on the root files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        rootFile = cmdLineUtils.openROOTFile(fileName)
        if not rootFile:
            retcode += 1
            continue

        # Fill the key list (almost the same as in rools)
        keyList = keyListExtended(rootFile,pathSplitList)
        for key in keyList:
            if cmdLineUtils.isTreeKey(key):
                obj = key.ReadObj()
                for branch in obj.GetListOfBranches():
                    if not outputOptionValue:
                        outputFileName = \
                            key.GetName() + "_" + branch.GetName() + "." +formatOptionValue
                        if directoryOptionValue:
                            outputFileName = os.path.join( \
                                directoryOptionValue,outputFileName)
                    obj.Draw(drawOptionValue)
                    if outputOptionValue or formatOptionValue == 'pdf':
                        objTitle = "Title:"+branch.GetName()+" : "+branch.GetTitle()
                        canvas.Print(outputFileName,objTitle)
                    else:
                        canvas.Print(outputFileName,formatOptionValue)
            else:
                if not outputOptionValue:
                    outputFileName = key.GetName() + "." +formatOptionValue
                    if directoryOptionValue:
                        outputFileName = os.path.join( \
                            directoryOptionValue,outputFileName)
                obj = key.ReadObj()
                obj.Draw(drawOptionValue)
                if outputOptionValue or formatOptionValue == 'pdf':
                    objTitle = "Title:"+key.GetClassName()+" : "+key.GetTitle()
                    canvas.Print(outputFileName,objTitle)
                else:
                    canvas.Print(outputFileName,formatOptionValue)
        rootFile.Close()

    # End to print (output option)
    if outputOptionValue:
        canvas.Print(outputFileName+"]",objTitle)

    return retcode

sys.exit(execute())

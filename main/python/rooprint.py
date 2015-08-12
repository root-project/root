#!/usr/bin/python

"""Command line to print ROOT files contents on ps,pdf or png,gif..."""

from cmdLineUtils import *

formatList = ["ps","eps","pdf","svg","tex","gif","xpm","png","jpg"]

# Help strings
COMMAND_HELP = \
    "Print ROOT files contents on ps,pdf or pictures files " + \
    "(for more informations please look at the man page)."
DIRECTORY_HELP = \
    "put output files in a subdirectory named DIRECTORY."
FORMAT_HELP = \
    "specify output format (ex: pdf, png)."
OUTPUT_HELP = \
    "merge files in a file named OUTPUT (only for ps and pdf)."

##### Beginning of the main code #####

# Collect arguments with the module argparse
parser = argparse.ArgumentParser(description=COMMAND_HELP)
parser.add_argument("sourcePatternList", help=SOURCES_HELP, nargs='+')
parser.add_argument("-d", "--directory", help=DIRECTORY_HELP)
parser.add_argument("-o", "--output", help=OUTPUT_HELP)
parser.add_argument("-f", "--format", help=FORMAT_HELP)
args = parser.parse_args()

# Create a list of tuples that contain source ROOT file names
# and lists of path in these files
sourceList = \
    [tup for pattern in args.sourcePatternList \
    for tup in patternToFileNameAndPathSplitList(pattern)]

# Create a dictionnary with options
optDict = vars(args)

# Initialize the canvas
ROOT.gErrorIgnoreLevel = 9999
ROOT.gROOT.SetBatch()
canvas = ROOT.TCanvas("canvas")

# Take the format of the output file (format option)
if not optDict["format"] and optDict["output"]:
    fileName = optDict["output"]
    fileFormat = fileName.split(".")[-1]
    if fileFormat in formatList: optDict["format"] = fileFormat

# Use pdf as default format
if not optDict["format"]: optDict["format"] = "pdf"

# Create the output directory (directory option)
if optDict["directory"]:
    if not os.path.isdir(os.path.join(os.getcwd(),optDict["directory"])):
        os.mkdir(optDict["directory"])

# Make the output name, begin to print (output option)
if optDict["output"]:
    if optDict["format"] in ['ps','pdf']:
        outputFileName = optDict["output"]
        if optDict["directory"]: outputFileName = \
           optDict["directory"] + "/" + outputFileName
        canvas.Print(outputFileName+"[",optDict["format"])
    else:
        print("Can't merge pictures, only postscript or pdf files")
        optDict["output"] = ""

# Loop on the root files
for fileName, pathSplitList in sourceList:
    with stderrRedirected():
        rootFile = ROOT.TFile.Open(fileName)
    # Fill the key list (almost the same as in rools)
    objList,dirList = keyClassSpliter(rootFile,pathSplitList)
    keyList = [getKey(rootFile,pathSplit) for pathSplit in objList]
    keyList.sort()
    dirList.sort()
    for pathSplit in dirList: keyList.extend(getKeyList(rootFile,pathSplit))
    keyList = [key for key in keyList if not isDirectoryKey(key)]
    for key in keyList:
        if isTreeKey(key):
            obj = key.ReadObj()
            for branch in obj.GetListOfBranches():
                if not optDict["output"]:
                    outputFileName = \
                        branch.GetName() + "." +optDict["format"]
                    if optDict["directory"]:
                        outputFileName = os.path.join( \
                            optDict["directory"],outputFileName)
                obj.Draw(branch.GetName())
                if optDict["output"] or optDict["format"] == 'pdf':
                    objTitle = "Title:"+branch.GetName()+" : "+branch.GetTitle()
                    canvas.Print(outputFileName,objTitle)
                else:
                    canvas.Print(outputFileName,optDict["format"])
        else:
            if not optDict["output"]:
                outputFileName = key.GetName() + "." +optDict["format"]
                if optDict["directory"]:
                    outputFileName = os.path.join( \
                        optDict["directory"],outputFileName)
            obj = key.ReadObj()
            obj.Draw()
            if optDict["output"] or optDict["format"] == 'pdf':
                objTitle = "Title:"+key.GetClassName()+" : "+key.GetTitle()
                canvas.Print(outputFileName,objTitle)
            else:
                canvas.Print(outputFileName,optDict["format"])
    rootFile.Close()

# End to print (output option)
if optDict["output"]:
    canvas.Print(outputFileName+"]",objTitle)

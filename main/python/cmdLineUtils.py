#!/usr/bin/env @python@

# ROOT command line tools module: cmdLineUtils
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 20/08/15

"""Contain utils for ROOT command line tools"""

##########
# Stream redirect functions
# The original code of the these functions can be found here :
# http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
# Thanks J.F. Sebastian !!

from contextlib import contextmanager
import os
import sys
from time import sleep
from itertools import zip_longest

def fileno(file_or_fd):
    """
    Look for 'fileno' attribute.
    """
    fd = getattr(file_or_fd, "fileno", lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextmanager
def streamRedirected(source=sys.stdout, destination=os.devnull):
    """
    Redirect the output from source to destination.
    """
    stdout_fd = fileno(source)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), "wb") as copied:
        source.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(destination), stdout_fd)  # $ exec >&destination
        except ValueError:  # filename
            with open(destination, "wb") as destination_file:
                os.dup2(destination_file.fileno(), stdout_fd)  # $ exec > destination
        try:
            yield source  # allow code to be run with the redirected stream
        finally:
            # restore source to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            source.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def stdoutRedirected():
    """
    Redirect the output from sys.stdout to os.devnull.
    """
    return streamRedirected(sys.stdout, os.devnull)


def stderrRedirected():
    """
    Redirect the output from sys.stderr to os.devnull.
    """
    return streamRedirected(sys.stderr, os.devnull)


# The end of streamRedirected functions
##########

##########
# Imports

##
# redirect output (escape characters during ROOT importation...)
with stdoutRedirected():
    import ROOT
# Silence Davix warning (see ROOT-7577)
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.GetVersion()

import argparse
import glob
import fnmatch
import logging

LOG_FORMAT = "%(levelname)s: %(message)s"
logging.basicConfig(format=LOG_FORMAT)

# The end of imports
##########

##########
# Different functions to get a parser of arguments and options


def _getParser(theHelp, theEpilog):
    """
    Get a commandline parser with the defaults of the commandline utils.
    """
    return argparse.ArgumentParser(
        description=theHelp, formatter_class=argparse.RawDescriptionHelpFormatter, epilog=theEpilog
    )


def getParserSingleFile(theHelp, theEpilog=""):
    """
    Get a commandline parser with the defaults of the commandline utils and a
    source file or not.
    """
    parser = _getParser(theHelp, theEpilog)
    parser.add_argument("FILE", nargs="?", help="Input file")
    return parser


def getParserFile(theHelp, theEpilog=""):
    """
    Get a commandline parser with the defaults of the commandline utils and a
    list of source files.
    """
    parser = _getParser(theHelp, theEpilog)
    parser.add_argument("FILE", nargs="+", help="Input file")
    return parser


def getParserSourceDest(theHelp, theEpilog=""):
    """
    Get a commandline parser with the defaults of the commandline utils,
    a list of source files and a destination file.
    """
    parser = _getParser(theHelp, theEpilog)
    parser.add_argument("SOURCE", nargs="+", help="Source file")
    parser.add_argument("DEST", help="Destination file")
    return parser


# The end of get parser functions
##########

##########
# Several utils


@contextmanager
def _setIgnoreLevel(level):
    originalLevel = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = level
    yield
    ROOT.gErrorIgnoreLevel = originalLevel


def changeDirectory(rootFile, pathSplit):
    """
    Change the current directory (ROOT.gDirectory) by the corresponding (rootFile,pathSplit)
    """
    rootFile.cd()
    for directoryName in pathSplit:
        theDir = ROOT.gDirectory.Get(directoryName)
        if not theDir:
            logging.warning("Directory %s does not exist." % directoryName)
            return 1
        else:
            theDir.cd()
    return 0


def createDirectory(rootFile, pathSplit):
    """
    Add a directory named 'pathSplit[-1]' in (rootFile,pathSplit[:-1])
    """
    retcode = changeDirectory(rootFile, pathSplit[:-1])
    if retcode == 0:
        ROOT.gDirectory.mkdir(pathSplit[-1])
    return retcode


def getFromDirectory(objName):
    """
    Get the object objName from the current directory
    """
    return ROOT.gDirectory.Get(objName)


def isExisting(rootFile, pathSplit):
    """
    Return True if the object, corresponding to (rootFile,pathSplit), exits
    """
    changeDirectory(rootFile, pathSplit[:-1])
    return ROOT.gDirectory.GetListOfKeys().Contains(pathSplit[-1])


def isDirectoryKey(key):
    """
    Return True if the object, corresponding to the key, inherits from TDirectory
    """
    import cppyy

    classname = key.GetClassName()
    cl = ROOT.gROOT.GetClass(classname)
    if cl == cppyy.nullptr:
        logging.warning("Unknown class to ROOT: " + classname)
        return False
    return cl.InheritsFrom(ROOT.TDirectory.Class())


def isTreeKey(key):
    """
    Return True if the object, corresponding to the key, inherits from TTree
    """
    import cppyy

    classname = key.GetClassName()
    cl = ROOT.gROOT.GetClass(classname)
    if cl == cppyy.nullptr:
        logging.warning("Unknown class to ROOT: " + classname)
        return False
    return cl.InheritsFrom(ROOT.TTree.Class())


def isTHnSparseKey(key):
    """
    Return True if the object, corresponding to the key, inherits from THnSparse
    """
    import cppyy

    classname = key.GetClassName()
    cl = ROOT.gROOT.GetClass(classname)
    if cl == cppyy.nullptr:
        logging.warning("Unknown class to ROOT: " + classname)
        return False
    return cl.InheritsFrom(ROOT.THnSparse.Class())


def getKey(rootFile, pathSplit):
    """
    Get the key of the corresponding object (rootFile,pathSplit)
    """
    changeDirectory(rootFile, pathSplit[:-1])
    return ROOT.gDirectory.GetKey(pathSplit[-1])


def isDirectory(rootFile, pathSplit):
    """
    Return True if the object, corresponding to (rootFile,pathSplit), inherits from TDirectory
    """
    if pathSplit == []:
        return True  # the object is the rootFile itself
    else:
        return isDirectoryKey(getKey(rootFile, pathSplit))


def isTree(rootFile, pathSplit):
    """
    Return True if the object, corresponding to (rootFile,pathSplit), inherits from TTree
    """
    if pathSplit == []:
        return False  # the object is the rootFile itself
    else:
        return isTreeKey(getKey(rootFile, pathSplit))


def getKeyList(rootFile, pathSplit):
    """
    Get the list of keys of the directory (rootFile,pathSplit),
    if (rootFile,pathSplit) is not a directory then get the key in a list
    """
    if isDirectory(rootFile, pathSplit):
        changeDirectory(rootFile, pathSplit)
        return ROOT.gDirectory.GetListOfKeys()
    else:
        return [getKey(rootFile, pathSplit)]


def keyListSort(keyList):
    """
    Sort list of keys by their names ignoring the case
    """
    keyList.sort(key=lambda x: x.GetName().lower())


def tupleListSort(tupleList):
    """
    Sort list of tuples by their first elements ignoring the case
    """
    tupleList.sort(key=lambda x: x[0].lower())


def dirListSort(dirList):
    """
    Sort list of directories by their names ignoring the case
    """
    dirList.sort(key=lambda x: [n.lower() for n in x])


def keyClassSplitter(rootFile, pathSplitList):
    """
    Return a list of directories and a list of keys corresponding
    to the other objects, for rootls and rootprint use
    """
    keyList = []
    dirList = []
    for pathSplit in pathSplitList:
        if pathSplit == []:
            dirList.append(pathSplit)
        elif isDirectory(rootFile, pathSplit):
            dirList.append(pathSplit)
        else:
            keyList.append(getKey(rootFile, pathSplit))
    keyListSort(keyList)
    dirListSort(dirList)
    return keyList, dirList


def openROOTFile(fileName, mode="read"):
    """
    Open the ROOT file corresponding to fileName in the corresponding mode,
    redirecting the output not to see missing dictionnaries

    Returns:
        theFile (TFile)
    """
    # with stderrRedirected():
    with _setIgnoreLevel(ROOT.kError):
        theFile = ROOT.TFile.Open(fileName, mode)
    if not theFile:
        logging.warning("File %s does not exist", fileName)
    return theFile


def openROOTFileCompress(fileName, compress, recreate):
    """
    Open a ROOT file (like openROOTFile) with the possibility
    to change compression settings
    """
    if compress != None and os.path.isfile(fileName):
        logging.warning("can't change compression settings on existing file")
        return None
    mode = "recreate" if recreate else "update"
    theFile = openROOTFile(fileName, mode)
    if compress != None:
        theFile.SetCompressionSettings(compress)
    return theFile


def joinPathSplit(pathSplit):
    """
    Join the pathSplit with '/'
    """
    return "/".join(pathSplit)


MANY_OCCURENCE_WARNING = "Several versions of '{0}' are present in '{1}'. Only the most recent will be considered."


def manyOccurenceRemove(pathSplitList, fileName):
    """
    Search for double occurence of the same pathSplit and remove them
    """
    if len(pathSplitList) > 1:
        for n in pathSplitList:
            if pathSplitList.count(n) != 1:
                logging.warning(MANY_OCCURENCE_WARNING.format(joinPathSplit(n), fileName))
                while n in pathSplitList and pathSplitList.count(n) != 1:
                    pathSplitList.remove(n)


def patternToPathSplitList(fileName, pattern):
    """
    Get the list of pathSplit of objects in the ROOT file
    corresponding to fileName that match with the pattern
    """
    # Open ROOT file
    rootFile = openROOTFile(fileName)
    if not rootFile:
        return []

    # Split pattern avoiding multiple slash problem
    patternSplit = [n for n in pattern.split("/") if n != ""]

    # Main loop
    pathSplitList = [[]]
    for patternPiece in patternSplit:
        newPathSplitList = []
        for pathSplit in pathSplitList:
            if isDirectory(rootFile, pathSplit):
                changeDirectory(rootFile, pathSplit)
                newPathSplitList.extend(
                    [
                        pathSplit + [key.GetName()]
                        for key in ROOT.gDirectory.GetListOfKeys()
                        if fnmatch.fnmatch(key.GetName(), patternPiece)
                    ]
                )
        pathSplitList = newPathSplitList

    # No match
    if pathSplitList == []:
        logging.warning("can't find {0} in {1}".format(pattern, fileName))

    # Same match (remove double occurrences from the list)
    manyOccurenceRemove(pathSplitList, fileName)

    return pathSplitList


def fileNameListMatch(filePattern, wildcards):
    """
    Get the list of fileName that match with objPattern
    """
    if wildcards:
        return [os.path.expandvars(os.path.expanduser(i)) for i in glob.iglob(filePattern)]
    else:
        return [os.path.expandvars(os.path.expanduser(filePattern))]


def pathSplitListMatch(fileName, objPattern, wildcards):
    """
    Get the list of pathSplit that match with objPattern
    """
    if wildcards:
        return patternToPathSplitList(fileName, objPattern)
    else:
        return [[n for n in objPattern.split("/") if n != ""]]


def patternToFileNameAndPathSplitList(pattern, wildcards=True):
    """
    Get the list of tuple containing both :
    - ROOT file name
    - list of splited path (in the corresponding file) of objects that matche
    Use unix wildcards by default
    """
    rootFilePattern = "*.root"
    rootObjPattern = rootFilePattern + ":*"
    httpRootFilePattern = "htt*://*.root"
    httpRootObjPattern = httpRootFilePattern + ":*"
    xrootdRootFilePattern = "root://*.root"
    xrootdRootObjPattern = xrootdRootFilePattern + ":*"
    s3RootFilePattern = "s3://*.root"
    s3RootObjPattern = s3RootFilePattern + ":*"
    gsRootFilePattern = "gs://*.root"
    gsRootObjPattern = gsRootFilePattern + ":*"
    pcmFilePattern = "*.pcm"
    pcmObjPattern = pcmFilePattern + ":*"

    if (
        fnmatch.fnmatch(pattern, httpRootObjPattern)
        or fnmatch.fnmatch(pattern, xrootdRootObjPattern)
        or fnmatch.fnmatch(pattern, s3RootObjPattern)
        or fnmatch.fnmatch(pattern, gsRootObjPattern)
    ):
        patternSplit = pattern.rsplit(":", 1)
        fileName = patternSplit[0]
        objPattern = patternSplit[1]
        pathSplitList = pathSplitListMatch(fileName, objPattern, wildcards)
        return [(fileName, pathSplitList)]

    if (
        fnmatch.fnmatch(pattern, httpRootFilePattern)
        or fnmatch.fnmatch(pattern, xrootdRootFilePattern)
        or fnmatch.fnmatch(pattern, s3RootFilePattern)
        or fnmatch.fnmatch(pattern, gsRootFilePattern)
    ):
        fileName = pattern
        pathSplitList = [[]]
        return [(fileName, pathSplitList)]

    if fnmatch.fnmatch(pattern, rootObjPattern) or fnmatch.fnmatch(pattern, pcmObjPattern):
        patternSplit = pattern.split(":")
        filePattern = patternSplit[0]
        objPattern = patternSplit[1]
        fileNameList = fileNameListMatch(filePattern, wildcards)
        return [(fileName, pathSplitListMatch(fileName, objPattern, wildcards)) for fileName in fileNameList]

    if fnmatch.fnmatch(pattern, rootFilePattern) or fnmatch.fnmatch(pattern, pcmFilePattern):
        filePattern = pattern
        fileNameList = fileNameListMatch(filePattern, wildcards)
        pathSplitList = [[]]
        return [(fileName, pathSplitList) for fileName in fileNameList]

    logging.warning("{0}: No such file (or extension not supported)".format(pattern))
    return []


# End of utils
##########

##########
# Set of functions to put the arguments in shape


def getArgs(parser):
    """
    Get arguments corresponding to parser.
    """
    return parser.parse_args()


def getSourceListArgs(parser, wildcards=True):
    """
    Create a list of tuples that contain source ROOT file names
    and lists of path in these files as well as the original arguments
    """
    args = getArgs(parser)
    inputFiles = []
    try:
        inputFiles = args.FILE
    except:
        inputFiles = args.SOURCE
    sourceList = [tup for pattern in inputFiles for tup in patternToFileNameAndPathSplitList(pattern, wildcards)]
    return sourceList, args


def getSourceListOptDict(parser, wildcards=True):
    """
    Get the list of tuples and the dictionary with options

    returns:
        sourceList: a list of tuples with one list element per file
                    the first tuple entry being the root file,
                    the second a list of subdirectories,
                        each being represented as a list itself with a string per level
                    e.g.
                    rootls tutorial/tmva/TMVA.root:Method_BDT/BDT turns into
                    [('tutorials/tmva/TMVA.root', [['Method_BDT','BDT']])]
        vars(args): a dictionary of matched options, e.g.
                    {'longListing': False,
                     'oneColumn': False,
                     'treeListing': False,
                     'recursiveListing'; False,
                     'FILE': ['tutorials/tmva/TMVA.root:Method_BDT/BDT']
                     }
    """
    sourceList, args = getSourceListArgs(parser, wildcards)
    if sourceList == []:
        logging.error("Input file(s) not found!")
    return sourceList, vars(args)


def getSourceDestListOptDict(parser, wildcards=True):
    """
    Get the list of tuples of sources, create destination name, destination pathSplit
    and the dictionary with options
    """
    sourceList, args = getSourceListArgs(parser, wildcards)
    destList = patternToFileNameAndPathSplitList(args.DEST, wildcards=False)
    if destList != []:
        destFileName, destPathSplitList = destList[0]
        destPathSplit = destPathSplitList[0]
    else:
        destFileName = ""
        destPathSplit = []
    return sourceList, destFileName, destPathSplit, vars(args)


# The end of the set of functions to put the arguments in shape
##########

##########
# Several functions shared by rootcp, rootmv and rootrm

TARGET_ERROR = "target '{0}' is not a directory"
OMITTING_ERROR = "{0} '{1}' will be copied but not its subdirectories (if any). Use the -r option if you need a recursive copy."
OVERWRITE_ERROR = "cannot overwrite non-directory '{0}' with directory '{1}'"


def copyRootObject(sourceFile, sourcePathSplit, destFile, destPathSplit, oneSource, recursive, replace):
    """
    Initialize the recursive function 'copyRootObjectRecursive', written to be as unix-like as possible
    """
    retcode = 0
    isMultipleInput = not (oneSource and sourcePathSplit != [])
    recursiveOption = recursive
    # Multiple input and un-existing or non-directory destination
    # TARGET_ERROR
    if (
        isMultipleInput
        and destPathSplit != []
        and not (isExisting(destFile, destPathSplit) and isDirectory(destFile, destPathSplit))
    ):
        logging.warning(TARGET_ERROR.format(destPathSplit[-1]))
        retcode += 1
    # Entire ROOT file or directory in input omitting "-r" option
    if not recursiveOption:
        if sourcePathSplit == []:
            logging.warning(OMITTING_ERROR.format("file", sourceFile.GetName()))
            retcode += 1
        elif isDirectory(sourceFile, sourcePathSplit):
            logging.warning(OMITTING_ERROR.format("directory", sourcePathSplit[-1]))
            retcode += 1
    # Run copyRootObjectRecursive function with the wish
    # to follow the unix copy behaviour
    if sourcePathSplit == []:
        retcode += copyRootObjectRecursive(sourceFile, sourcePathSplit, destFile, destPathSplit, replace)
    else:
        setName = ""
        if not isMultipleInput and (destPathSplit != [] and not isExisting(destFile, destPathSplit)):
            setName = destPathSplit[-1]
        objectName = sourcePathSplit[-1]
        if isDirectory(sourceFile, sourcePathSplit):
            if setName != "":
                createDirectory(destFile, destPathSplit[:-1] + [setName])
                retcode += copyRootObjectRecursive(
                    sourceFile, sourcePathSplit, destFile, destPathSplit[:-1] + [setName], replace
                )
            elif isDirectory(destFile, destPathSplit):
                if not isExisting(destFile, destPathSplit + [objectName]):
                    createDirectory(destFile, destPathSplit + [objectName])
                if isDirectory(destFile, destPathSplit + [objectName]):
                    retcode += copyRootObjectRecursive(
                        sourceFile, sourcePathSplit, destFile, destPathSplit + [objectName], replace
                    )
                else:
                    logging.warning(OVERWRITE_ERROR.format(objectName, objectName))
                    retcode += 1
            else:
                logging.warning(OVERWRITE_ERROR.format(destPathSplit[-1], objectName))
                retcode += 1
        else:
            if setName != "":
                retcode += copyRootObjectRecursive(
                    sourceFile, sourcePathSplit, destFile, destPathSplit[:-1], replace, setName
                )
            elif isDirectory(destFile, destPathSplit):
                retcode += copyRootObjectRecursive(sourceFile, sourcePathSplit, destFile, destPathSplit, replace)
            else:
                setName = destPathSplit[-1]
                retcode += copyRootObjectRecursive(
                    sourceFile, sourcePathSplit, destFile, destPathSplit[:-1], replace, setName
                )
    return retcode


DELETE_ERROR = "object {0} was not existing, so it is not deleted"


def deleteObject(rootFile, pathSplit):
    """
    Delete the object 'pathSplit[-1]' from (rootFile,pathSplit[:-1])
    """
    retcode = changeDirectory(rootFile, pathSplit[:-1])
    if retcode == 0:
        fileName = pathSplit[-1]
        if isExisting(rootFile, pathSplit):
            ROOT.gDirectory.Delete(fileName + ";*")
        else:
            logging.warning(DELETE_ERROR.format(fileName))
            retcode += 1
    return retcode


def copyRootObjectRecursive(sourceFile, sourcePathSplit, destFile, destPathSplit, replace, setName=""):
    """
    Copy objects from a file or directory (sourceFile,sourcePathSplit)
    to an other file or directory (destFile,destPathSplit)
    - Has the will to be unix-like
    - that's a recursive function
    - Python adaptation of a root input/output tutorial : copyFiles.C
    """
    retcode = 0
    replaceOption = replace
    seen = {}
    for key in getKeyList(sourceFile, sourcePathSplit):
        objectName = key.GetName()

        # write keys only if the cycle is higher than before
        if objectName not in seen.keys():
            seen[objectName] = key
        else:
            if seen[objectName].GetCycle() < key.GetCycle():
                seen[objectName] = key
            else:
                continue

        if isDirectoryKey(key):
            if not isExisting(destFile, destPathSplit + [objectName]):
                createDirectory(destFile, destPathSplit + [objectName])
            if isDirectory(destFile, destPathSplit + [objectName]):
                retcode += copyRootObjectRecursive(
                    sourceFile, sourcePathSplit + [objectName], destFile, destPathSplit + [objectName], replace
                )
            else:
                logging.warning(OVERWRITE_ERROR.format(objectName, objectName))
                retcode += 1
        elif isTreeKey(key):
            T = key.GetMotherDir().Get(objectName + ";" + str(key.GetCycle()))
            if replaceOption and isExisting(destFile, destPathSplit + [T.GetName()]):
                retcodeTemp = deleteObject(destFile, destPathSplit + [T.GetName()])
                if retcodeTemp:
                    retcode += retcodeTemp
                    continue
            changeDirectory(destFile, destPathSplit)
            newT = T.CloneTree(-1, "fast")
            if setName != "":
                newT.SetName(setName)
            newT.Write()
        else:
            obj = key.ReadObj()
            if replaceOption and isExisting(destFile, destPathSplit + [setName]):
                changeDirectory(destFile, destPathSplit)
                otherObj = getFromDirectory(setName)
                retcodeTemp = deleteObject(destFile, destPathSplit + [setName])
                if retcodeTemp:
                    retcode += retcodeTemp
                    continue
                else:
                    if isinstance(obj, ROOT.TNamed):
                        obj.SetName(setName)
                    changeDirectory(destFile, destPathSplit)
                    obj.Write()
            elif issubclass(obj.__class__, ROOT.TCollection):
                # probably the object was written with kSingleKey
                changeDirectory(destFile, destPathSplit)
                obj.Write(setName, ROOT.TObject.kSingleKey)
            else:
                if setName != "":
                    if isinstance(obj, ROOT.TNamed):
                        obj.SetName(setName)
                else:
                    if isinstance(obj, ROOT.TNamed):
                        obj.SetName(objectName)
                changeDirectory(destFile, destPathSplit)
                obj.Write()
            obj.Delete()
    changeDirectory(destFile, destPathSplit)
    ROOT.gDirectory.SaveSelf(ROOT.kTRUE)
    return retcode


FILE_REMOVE_ERROR = "cannot remove '{0}': Is a ROOT file"
DIRECTORY_REMOVE_ERROR = "cannot remove '{0}': Is a directory"
ASK_FILE_REMOVE = "remove '{0}' ? (y/n) : "
ASK_OBJECT_REMOVE = "remove '{0}' from '{1}' ? (y/n) : "


def deleteRootObject(rootFile, pathSplit, interactive, recursive):
    """
    Remove the object (rootFile,pathSplit)
    -interactive : prompt before every removal
    -recursive : allow directory, and ROOT file, removal
    """
    retcode = 0
    if not recursive and isDirectory(rootFile, pathSplit):
        if pathSplit == []:
            logging.warning(FILE_REMOVE_ERROR.format(rootFile.GetName()))
            retcode += 1
        else:
            logging.warning(DIRECTORY_REMOVE_ERROR.format(pathSplit[-1]))
            retcode += 1
    else:
        if interactive:
            if pathSplit != []:
                answer = input(ASK_OBJECT_REMOVE.format("/".join(pathSplit), rootFile.GetName()))
            else:
                answer = input(ASK_FILE_REMOVE.format(rootFile.GetName()))
            remove = answer.lower() == "y"
        else:
            remove = True
        if remove:
            if pathSplit != []:
                retcode += deleteObject(rootFile, pathSplit)
            else:
                rootFile.Close()
                os.remove(rootFile.GetName())
    return retcode


# End of functions shared by rootcp, rootmv and rootrm
##########

##########
# Help strings for ROOT command line tools

# Arguments
SOURCE_HELP = "path of the source."
SOURCES_HELP = "path of the source(s)."
DEST_HELP = "path of the destination."

# Options
COMPRESS_HELP = """change the compression settings of the
destination file (if not already existing)."""
INTERACTIVE_HELP = "prompt before every removal."
RECREATE_HELP = "recreate the destination file."
RECURSIVE_HELP = "recurse inside directories"
REPLACE_HELP = "replace object if already existing"

# End of help strings
##########

##########
# ROOTBROWSE


def _openBrowser(rootFile=None):
    browser = ROOT.TBrowser()
    if ROOT.gSystem.InheritsFrom("TMacOSXSystem") or browser.IsWeb():
        print("Press ctrl+c to exit.")
        try:
            while True:
                if ROOT.gROOT.IsInterrupted() or ROOT.gSystem.ProcessEvents():
                    break
                sleep(0.01)
        except (KeyboardInterrupt, SystemExit):
            pass
    else:
        input("Press enter to exit.")


def rootBrowse(fileName=None):
    if fileName:
        rootFile = openROOTFile(fileName)
        if not rootFile:
            return 1
        _openBrowser(rootFile)
        rootFile.Close()
    else:
        _openBrowser()
    return 0


# End of ROOTBROWSE
##########

##########
# ROOTCP


def _copyObjects(fileName, pathSplitList, destFile, destPathSplit, oneFile, recursive, replace):
    retcode = 0
    destFileName = destFile.GetName()
    rootFile = openROOTFile(fileName) if fileName != destFileName else destFile
    if not rootFile:
        return 1
    ROOT.gROOT.GetListOfFiles().Remove(rootFile)  # Fast copy necessity
    for pathSplit in pathSplitList:
        oneSource = oneFile and len(pathSplitList) == 1
        retcode += copyRootObject(rootFile, pathSplit, destFile, destPathSplit, oneSource, recursive, replace)
    if fileName != destFileName:
        rootFile.Close()
    return retcode


def rootCp(sourceList, destFileName, destPathSplit, compress=None, recreate=False, recursive=False, replace=False):
    # Check arguments
    if sourceList == [] or destFileName == "":
        return 1
    if recreate and destFileName in [n[0] for n in sourceList]:
        logging.error("cannot recreate destination file if this is also a source file")
        return 1

    # Open destination file
    destFile = openROOTFileCompress(destFileName, compress, recreate)
    if not destFile:
        return 1
    ROOT.gROOT.GetListOfFiles().Remove(destFile)  # Fast copy necessity

    # Loop on the root files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += _copyObjects(
            fileName, pathSplitList, destFile, destPathSplit, len(sourceList) == 1, recursive, replace
        )
    destFile.Close()
    return retcode


# End of ROOTCP
##########

##########
# ROOTEVENTSELECTOR


def _setBranchStatus(tree, branchSelectionString, status=0):
    """This is used by _copyTreeSubset() to turn on/off branches"""
    for branchToModify in branchSelectionString.split(","):
        logging.info("Setting branch status to %d for %s" % (status, branchToModify))
        tree.SetBranchStatus(branchToModify, status)
    return tree


def _copyTreeSubset(
    sourceFile,
    sourcePathSplit,
    destFile,
    destPathSplit,
    firstEvent,
    lastEvent,
    selectionString,
    branchinclude,
    branchexclude,
):
    """Copy a subset of the tree from (sourceFile,sourcePathSplit)
    to (destFile,destPathSplit) according to options in optDict"""
    retcode = changeDirectory(sourceFile, sourcePathSplit[:-1])
    if retcode != 0:
        return retcode
    bigTree = getFromDirectory(sourcePathSplit[-1])
    nbrEntries = bigTree.GetEntries()
    # changeDirectory for the small tree not to be memory-resident
    retcode = changeDirectory(destFile, destPathSplit)
    if retcode != 0:
        return retcode

    if lastEvent == -1:
        lastEvent = nbrEntries - 1
    numberOfEntries = (lastEvent - firstEvent) + 1

    # "Slim" tree by removing branches -
    # This is done after the skimming to allow for the user to skim on a
    # branch they no longer need to keep
    outputTree = bigTree
    if branchexclude:
        _setBranchStatus(outputTree, branchexclude, 0)
    if branchinclude:
        _setBranchStatus(outputTree, branchinclude, 1)
    if branchexclude or branchinclude:
        outputTree = outputTree.CloneTree()

    # "Skim" events based on branch values using selectionString
    # as well as selecting a range of events by index
    outputTree = outputTree.CopyTree(selectionString, "", numberOfEntries, firstEvent)

    outputTree.Write()
    return retcode


def _copyTreeSubsets(
    fileName, pathSplitList, destFile, destPathSplit, first, last, selectionString, branchinclude, branchexclude
):
    retcode = 0
    destFileName = destFile.GetName()
    rootFile = openROOTFile(fileName) if fileName != destFileName else destFile
    if not rootFile:
        return 1
    for pathSplit in pathSplitList:
        if isTree(rootFile, pathSplit):
            retcode += _copyTreeSubset(
                rootFile, pathSplit, destFile, destPathSplit, first, last, selectionString, branchinclude, branchexclude
            )
    if fileName != destFileName:
        rootFile.Close()
    return retcode


def rootEventselector(
    sourceList,
    destFileName,
    destPathSplit,
    compress=None,
    recreate=False,
    first=0,
    last=-1,
    selectionString="",
    branchinclude="",
    branchexclude="",
):
    # Check arguments
    if sourceList == [] or destFileName == "":
        return 1
    if recreate and destFileName in sourceList:
        logging.error("cannot recreate destination file if this is also a source file")
        return 1

    # Open destination file
    destFile = openROOTFileCompress(destFileName, compress, recreate)
    if not destFile:
        return 1

    # Loop on the root file
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += _copyTreeSubsets(
            fileName, pathSplitList, destFile, destPathSplit, first, last, selectionString, branchinclude, branchexclude
        )
    destFile.Close()
    return retcode


# End of ROOTEVENTSELECTOR
##########

##########
# ROOTMKDIR

MKDIR_ERROR = "cannot create directory '{0}'"


def _createDirectories(rootFile, pathSplit, parents):
    """Same behaviour as createDirectory but allows the possibility
    to build an whole path recursively with the option \"parents\" """
    retcode = 0
    lenPathSplit = len(pathSplit)
    if lenPathSplit == 0:
        pass
    elif parents:
        for i in range(lenPathSplit):
            currentPathSplit = pathSplit[: i + 1]
            if not (isExisting(rootFile, currentPathSplit) and isDirectory(rootFile, currentPathSplit)):
                retcode += createDirectory(rootFile, currentPathSplit)
    else:
        doMkdir = True
        for i in range(lenPathSplit - 1):
            currentPathSplit = pathSplit[: i + 1]
            if not (isExisting(rootFile, currentPathSplit) and isDirectory(rootFile, currentPathSplit)):
                doMkdir = False
                break
        if doMkdir:
            retcode += createDirectory(rootFile, pathSplit)
        else:
            logging.warning(MKDIR_ERROR.format("/".join(pathSplit)))
            retcode += 1
    return retcode


def _rootMkdirProcessFile(fileName, pathSplitList, parents):
    retcode = 0
    rootFile = openROOTFile(fileName, "update")
    if not rootFile:
        return 1
    for pathSplit in pathSplitList:
        retcode += _createDirectories(rootFile, pathSplit, parents)
    rootFile.Close()
    return retcode


def rootMkdir(sourceList, parents=False):
    # Check arguments
    if sourceList == []:
        return 1

    # Loop on the ROOT files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += _rootMkdirProcessFile(fileName, pathSplitList, parents)
    return retcode


# End of ROOTMKDIR
##########

##########
# ROOTMV

MOVE_ERROR = "error during copy of {0}, it is not removed from {1}"


def _moveObjects(fileName, pathSplitList, destFile, destPathSplit, oneFile, interactive):
    retcode = 0
    recursive = True
    replace = True
    destFileName = destFile.GetName()
    rootFile = openROOTFile(fileName, "update") if fileName != destFileName else destFile
    if not rootFile:
        return 1
    ROOT.gROOT.GetListOfFiles().Remove(rootFile)  # Fast copy necessity
    for pathSplit in pathSplitList:
        oneSource = oneFile and len(pathSplitList) == 1
        retcodeTemp = copyRootObject(rootFile, pathSplit, destFile, destPathSplit, oneSource, recursive, replace)
        if not retcodeTemp:
            retcode += deleteRootObject(rootFile, pathSplit, interactive, recursive)
        else:
            logging.warning(MOVE_ERROR.format("/".join(pathSplit), rootFile.GetName()))
            retcode += retcodeTemp
    if fileName != destFileName:
        rootFile.Close()
    return retcode


def rootMv(sourceList, destFileName, destPathSplit, compress=None, interactive=False, recreate=False):
    # Check arguments
    if sourceList == [] or destFileName == "":
        return 1
    if recreate and destFileName in sourceList:
        logging.error("cannot recreate destination file if this is also a source file")
        return 1

    # Open destination file
    destFile = openROOTFileCompress(destFileName, compress, recreate)
    if not destFile:
        return 1
    ROOT.gROOT.GetListOfFiles().Remove(destFile)  # Fast copy necessity

    # Loop on the root files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += _moveObjects(fileName, pathSplitList, destFile, destPathSplit, len(sourceList) == 1, interactive)
    destFile.Close()
    return retcode


# End of ROOTMV
##########

##########
# ROOTPRINT


def _keyListExtended(rootFile, pathSplitList):
    keyList, dirList = keyClassSplitter(rootFile, pathSplitList)
    for pathSplit in dirList:
        keyList.extend(getKeyList(rootFile, pathSplit))
    keyList = [key for key in keyList if not isDirectoryKey(key)]
    keyListSort(keyList)
    return keyList


def rootPrint(
    sourceList,
    directoryOption=None,
    divideOption=None,
    drawOption="",
    formatOption=None,
    outputOption=None,
    sizeOption=None,
    styleOption=None,
    verboseOption=False,
):
    # Check arguments
    if sourceList == []:
        return 1
    tupleListSort(sourceList)

    # Don't open windows
    ROOT.gROOT.SetBatch()

    # (Style option)
    if styleOption:
        ROOT.gInterpreter.ProcessLine(".x {0}".format(styleOption))

    # (Verbose option)
    if not verboseOption:
        ROOT.gErrorIgnoreLevel = 9999

    # Initialize the canvas (Size option)
    if sizeOption:
        try:
            width, height = sizeOption.split("x")
            width = int(width)
            height = int(height)
        except ValueError:
            logging.warning("canvas size is on a wrong format")
            return 1
        canvas = ROOT.TCanvas("canvas", "canvas", width, height)
    else:
        canvas = ROOT.TCanvas("canvas")

    # Divide the canvas (Divide option)
    if divideOption:
        try:
            x, y = divideOption.split(",")
            x = int(x)
            y = int(y)
        except ValueError:
            logging.warning("divide is on a wrong format")
            return 1
        canvas.Divide(x, y)
        caseNumber = x * y

    # Take the format of the output file (formatOutput option)
    if not formatOption and outputOption:
        fileName = outputOption
        fileFormat = fileName.split(".")[-1]
        formatOption = fileFormat

    # Use pdf as default format
    if not formatOption:
        formatOption = "pdf"

    # Create the output directory (directory option)
    if directoryOption:
        if not os.path.isdir(os.path.join(os.getcwd(), directoryOption)):
            os.mkdir(directoryOption)

    # Make the output name, begin to print (output option)
    if outputOption:
        if formatOption in ["ps", "pdf"]:
            outputFileName = outputOption
            if directoryOption:
                outputFileName = directoryOption + "/" + outputFileName
            canvas.Print(outputFileName + "[", formatOption)
        else:
            logging.warning("can't merge pictures, only postscript or pdf files")
            return 1

    # Loop on the root files
    retcode = 0
    objDrawnNumber = 0
    openRootFiles = []
    for fileName, pathSplitList in sourceList:
        rootFile = openROOTFile(fileName)
        if not rootFile:
            retcode += 1
            continue
        openRootFiles.append(rootFile)
        # Fill the key list (almost the same as in root)
        keyList = _keyListExtended(rootFile, pathSplitList)
        for key in keyList:
            if isTreeKey(key):
                pass
            else:
                if divideOption:
                    canvas.cd(objDrawnNumber % caseNumber + 1)
                    objDrawnNumber += 1
                obj = key.ReadObj()
                obj.Draw(drawOption)
                if divideOption:
                    if objDrawnNumber % caseNumber == 0:
                        if not outputOption:
                            outputFileName = str(objDrawnNumber // caseNumber) + "." + formatOption
                            if directoryOption:
                                outputFileName = os.path.join(directoryOption, outputFileName)
                        canvas.Print(outputFileName, formatOption)
                        canvas.Clear()
                        canvas.Divide(x, y)
                else:
                    if not outputOption:
                        outputFileName = key.GetName() + "." + formatOption
                        if directoryOption:
                            outputFileName = os.path.join(directoryOption, outputFileName)
                    if outputOption or formatOption == "pdf":
                        objTitle = "Title:" + key.GetClassName() + " : " + key.GetTitle()
                        canvas.Print(outputFileName, objTitle)
                    else:
                        canvas.Print(outputFileName, formatOption)

    # Last page (divideOption)
    if divideOption:
        if objDrawnNumber % caseNumber != 0:
            if not outputOption:
                outputFileName = str(objDrawnNumber // caseNumber + 1) + "." + formatOption
                if directoryOption:
                    outputFileName = os.path.join(directoryOption, outputFileName)
            canvas.Print(outputFileName, formatOption)

    # End to print (output option)
    if outputOption:
        if not divideOption:
            canvas.Print(outputFileName + "]", objTitle)
        else:
            canvas.Print(outputFileName + "]")

    # Close ROOT files
    map(lambda rootFile: rootFile.Close(), openRootFiles)

    return retcode


# End of ROOTPRINT
##########

##########
# ROOTRM


def _removeObjects(fileName, pathSplitList, interactive=False, recursive=False):
    retcode = 0
    rootFile = openROOTFile(fileName, "update")
    if not rootFile:
        return 1
    for pathSplit in pathSplitList:
        retcode += deleteRootObject(rootFile, pathSplit, interactive, recursive)
    rootFile.Close()
    return retcode


def rootRm(sourceList, interactive=False, recursive=False):
    # Check arguments
    if sourceList == []:
        return 1

    # Loop on the root files
    retcode = 0
    for fileName, pathSplitList in sourceList:
        retcode += _removeObjects(fileName, pathSplitList, interactive, recursive)
    return retcode


# End of ROOTRM
##########

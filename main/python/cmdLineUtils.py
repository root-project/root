#!/usr/bin/python

# ROOT command line tools module: cmdLineUtils
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Contain utils for ROOT command line tools"""

##########
# Stream redirect functions
# The original code of the these functions can be found here :
# http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
# Thanks J.F. Sebastian !!

from contextlib import contextmanager
import os
import sys

def fileno(file_or_fd):
    """
    Look for 'fileno' attribute.
    """
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
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
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        source.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(destination), stdout_fd)  # $ exec >&destination
        except ValueError:  # filename
            with open(destination, 'wb') as destination_file:
                os.dup2(destination_file.fileno(), stdout_fd)  # $ exec > destination
        try:
            yield source # allow code to be run with the redirected stream
        finally:
            # restore source to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
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
# The gymnastic with sys argv  is necessary to workaround for ROOT-7577
argvTmp = sys.argv[:]
sys.argv = []
with stdoutRedirected():
    import ROOT
ROOT.gROOT.GetVersion()
sys.argv = argvTmp

import argparse
import glob
import fnmatch
import logging

# The end of imports
##########

##########
# Different functions to get a parser of arguments and options

def _getParser(theHelp, theEpilog):
   """
   Get a commandline parser with the defaults of the commandline utils.
   """
   return argparse.ArgumentParser(description=theHelp,
                                  formatter_class=argparse.RawDescriptionHelpFormatter,
                                  epilog = theEpilog)

def getParserSingleFile(theHelp, theEpilog=""):
   """
   Get a commandline parser with the defaults of the commandline utils and a
   source file or not.
   """
   parser = _getParser(theHelp, theEpilog)
   parser.add_argument("FILE", nargs='?', help="Input file")
   return parser

def getParserFile(theHelp, theEpilog=""):
   """
   Get a commandline parser with the defaults of the commandline utils and a
   list of source files.
   """
   parser = _getParser(theHelp, theEpilog)
   parser.add_argument("FILE", nargs='+', help="Input file")
   return parser

def getParserSourceDest(theHelp, theEpilog=""):
   """
   Get a commandline parser with the defaults of the commandline utils,
   a list of source files and a destination file.
   """
   parser = _getParser(theHelp, theEpilog)
   parser.add_argument("SOURCE", nargs='+', help="Source file")
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

def changeDirectory(rootFile,pathSplit):
    """
    Change the current directory (ROOT.gDirectory) by the corresponding (rootFile,pathSplit)
    """
    rootFile.cd()
    for directoryName in pathSplit:
        theDir = ROOT.gDirectory.Get(directoryName)
        if not theDir:
            logging.warning("Directory %s does not exist." %directoryName)
            return 1
        else:
            theDir.cd()
    return 0

def createDirectory(rootFile,pathSplit):
    """
    Add a directory named 'pathSplit[-1]' in (rootFile,pathSplit[:-1])
    """
    retcode = changeDirectory(rootFile,pathSplit[:-1])
    if retcode == 0: ROOT.gDirectory.mkdir(pathSplit[-1])
    return retcode

def getFromDirectory(objName):
    """
    Get the object objName from the current directory
    """
    return ROOT.gDirectory.Get(objName)

def isExisting(rootFile,pathSplit):
    """
    Return True if the object, corresponding to (rootFile,pathSplit), exits
    """
    changeDirectory(rootFile,pathSplit[:-1])
    return ROOT.gDirectory.GetListOfKeys().Contains(pathSplit[-1])

def isDirectoryKey(key):
    """
    Return True if the object, corresponding to the key, inherits from TDirectory
    """
    classname = key.GetClassName()
    cl = ROOT.gROOT.GetClass(classname)
    return cl.InheritsFrom(ROOT.TDirectory.Class())

def isTreeKey(key):
    """
    Return True if the object, corresponding to the key, inherits from TTree
    """
    classname = key.GetClassName()
    cl = ROOT.gROOT.GetClass(classname)
    return cl.InheritsFrom(ROOT.TTree.Class())

def getKey(rootFile,pathSplit):
    """
    Get the key of the corresponding object (rootFile,pathSplit)
    """
    changeDirectory(rootFile,pathSplit[:-1])
    return ROOT.gDirectory.GetKey(pathSplit[-1])

def isDirectory(rootFile,pathSplit):
    """
    Return True if the object, corresponding to (rootFile,pathSplit), inherits from TDirectory
    """
    if pathSplit == []: return True # the object is the rootFile itself
    else: return isDirectoryKey(getKey(rootFile,pathSplit))

def isTree(rootFile,pathSplit):
    """
    Return True if the object, corresponding to (rootFile,pathSplit), inherits from TTree
    """
    if pathSplit == []: return False # the object is the rootFile itself
    else: return isTreeKey(getKey(rootFile,pathSplit))

def getKeyList(rootFile,pathSplit):
    """
    Get the list of keys of the directory (rootFile,pathSplit),
    if (rootFile,pathSplit) is not a directory then get the key in a list
    """
    if isDirectory(rootFile,pathSplit):
        changeDirectory(rootFile,pathSplit)
        return ROOT.gDirectory.GetListOfKeys()
    else: return [getKey(rootFile,pathSplit)]

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

def keyClassSpliter(rootFile,pathSplitList):
    """
    Return a list of directories and a list of keys corresponding
    to the other objects, for rools and rooprint use
    """
    keyList = []
    dirList = []
    for pathSplit in pathSplitList:
        if pathSplit == []: dirList.append(pathSplit)
        elif isDirectory(rootFile,pathSplit): dirList.append(pathSplit)
        else: keyList.append(getKey(rootFile,pathSplit))
    keyListSort(keyList)
    dirListSort(dirList)
    return keyList,dirList

def openROOTFile(fileName, mode="read"):
    """
    Open the ROOT file corresponding to fileName in the corresponding mode,
    redirecting the output not to see missing dictionnaries
    """
    #with stderrRedirected():
    with _setIgnoreLevel(ROOT.kError):
        theFile = ROOT.TFile.Open(fileName, mode)
    if not theFile:
        logging.warning("File %s does not exist", fileName)
    return theFile

def openROOTFileCompress(fileName, optDict):
    """
    Open a ROOT file (like openROOTFile) with the possibility
    to change compression settings
    """
    compressOptionValue = optDict["compress"]
    if compressOptionValue != None and os.path.isfile(fileName):
        logging.warning("can't change compression settings on existing file")
        return None
    mode = "recreate" if optDict["recreate"] else "update"
    theFile = openROOTFile(fileName, mode)
    if compressOptionValue != None: theFile.SetCompressionSettings(compressOptionValue)
    return theFile

def joinPathSplit(pathSplit):
    """
    Join the pathSplit with '/'
    """
    return "/".join(pathSplit)

MANY_OCCURENCE_WARNING = "Same name objects aren't supported: '{0}' of '{1}' won't be processed"

def manyOccurenceRemove(pathSplitList,fileName):
    """
    Search for double occurence of the same pathSplit and remove them
    """
    if len(pathSplitList) > 1:
        for n in pathSplitList:
            if pathSplitList.count(n) != 1:
                logging.warning(MANY_OCCURENCE_WARNING.format(joinPathSplit(n),fileName))
                while n in pathSplitList: pathSplitList.remove(n)

def patternToPathSplitList(fileName,pattern):
    """
    Get the list of pathSplit of objects in the ROOT file
    corresponding to fileName that match with the pattern
    """
    # Open ROOT file
    rootFile = openROOTFile(fileName)
    if not rootFile: return []

    # Split pattern avoiding multiple slash problem
    patternSplit = [n for n in pattern.split("/") if n != ""]

    # Main loop
    pathSplitList = [[]]
    for patternPiece in patternSplit:
        newPathSplitList = []
        for pathSplit in pathSplitList:
            if isDirectory(rootFile,pathSplit):
                changeDirectory(rootFile,pathSplit)
                newPathSplitList.extend( \
                    [pathSplit + [key.GetName()] \
                    for key in ROOT.gDirectory.GetListOfKeys() \
                    if fnmatch.fnmatch(key.GetName(),patternPiece)])
        pathSplitList = newPathSplitList

    # No match
    if pathSplitList == []:
        logging.warning("can't find {0} in {1}".format(pattern,fileName))

    # Same match (remove double occurences from the list)
    manyOccurenceRemove(pathSplitList,fileName)

    return pathSplitList

def fileNameListMatch(filePattern,wildcards):
    """
    Get the list of fileName that match with objPattern
    """
    if wildcards: return [os.path.expandvars(os.path.expanduser(i)) for i in glob.iglob(filePattern)]
    else: return [os.path.expandvars(os.path.expanduser(filePattern))]

def pathSplitListMatch(fileName,objPattern,wildcards):
    """
    Get the list of pathSplit that match with objPattern
    """
    if wildcards: return patternToPathSplitList(fileName,objPattern)
    else: return [[n for n in objPattern.split("/") if n != ""]]

def patternToFileNameAndPathSplitList(pattern,wildcards = True):
    """
    Get the list of tuple containing both :
    - ROOT file name
    - list of splited path (in the corresponding file) of objects that matche
    Use unix wildcards by default
    """
    rootFilePattern = "*.root"
    rootObjPattern = rootFilePattern+":*"
    httpRootFilePattern = "htt*://*.root"
    httpRootObjPattern = httpRootFilePattern+":*"
    xrootdRootFilePattern = "root://*.root"
    xrootdRootObjPattern = xrootdRootFilePattern+":*"
    s3RootFilePattern = "s3://*.root"
    s3RootObjPattern = s3RootFilePattern+":*"
    gsRootFilePattern = "gs://*.root"
    gsRootObjPattern = gsRootFilePattern+":*"
    rfioRootFilePattern = "rfio://*.root"
    rfioRootObjPattern = rfioRootFilePattern+":*"
    pcmFilePattern = "*.pcm"
    pcmObjPattern = pcmFilePattern+":*"

    if fnmatch.fnmatch(pattern,httpRootObjPattern) or \
       fnmatch.fnmatch(pattern,xrootdRootObjPattern) or \
       fnmatch.fnmatch(pattern,s3RootObjPattern) or \
       fnmatch.fnmatch(pattern,gsRootObjPattern) or \
       fnmatch.fnmatch(pattern,rfioRootObjPattern):
        patternSplit = pattern.rsplit(":", 1)
        fileName = patternSplit[0]
        objPattern = patternSplit[1]
        pathSplitList = pathSplitListMatch(fileName,objPattern,wildcards)
        return [(fileName,pathSplitList)]

    if fnmatch.fnmatch(pattern,httpRootFilePattern) or \
       fnmatch.fnmatch(pattern,xrootdRootFilePattern) or \
       fnmatch.fnmatch(pattern,s3RootFilePattern) or \
       fnmatch.fnmatch(pattern,gsRootFilePattern) or \
       fnmatch.fnmatch(pattern,rfioRootFilePattern):
        fileName = pattern
        pathSplitList = [[]]
        return [(fileName,pathSplitList)]

    if fnmatch.fnmatch(pattern,rootObjPattern) or \
       fnmatch.fnmatch(pattern,pcmObjPattern):
        patternSplit = pattern.split(":")
        filePattern = patternSplit[0]
        objPattern = patternSplit[1]
        fileNameList = fileNameListMatch(filePattern,wildcards)
        return [(fileName,pathSplitListMatch(fileName,objPattern,wildcards)) for fileName in fileNameList]

    if fnmatch.fnmatch(pattern,rootFilePattern) or \
       fnmatch.fnmatch(pattern,pcmFilePattern):
        filePattern = pattern
        fileNameList = fileNameListMatch(filePattern,wildcards)
        pathSplitList = [[]]
        return [(fileName,pathSplitList) for fileName in fileNameList]

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

def getSourceListArgs(parser, wildcards = True):
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
   sourceList = \
      [tup for pattern in inputFiles \
      for tup in patternToFileNameAndPathSplitList(pattern,wildcards)]
   return sourceList, args

def getSourceListOptDict(parser, wildcards = True):
    """
    Get the list of tuples and the dictionary with options
    """
    sourceList, args = getSourceListArgs(parser, wildcards)
    return sourceList, vars(args)

def getSourceDestListOptDict(parser, wildcards = True):
    """
    Get the list of tuples of sources, create destination name, destination pathSplit
    and the dictionary with options
    """
    sourceList, args = getSourceListArgs(parser, wildcards)
    destList = \
        patternToFileNameAndPathSplitList( \
        args.DEST,wildcards=False)
    if destList != []:
        destFileName,destPathSplitList = destList[0]
        destPathSplit = destPathSplitList[0]
    else:
        destFileName = ""
        destPathSplit = []
    return sourceList, destFileName, destPathSplit, vars(args)

# The end of the set of functions to put the arguments in shape
##########

##########
# Several functions shared by roocp, roomv and roorm

TARGET_ERROR = "target '{0}' is not a directory"
OMITTING_FILE_ERROR = "omitting file '{0}'"
OMITTING_DIRECTORY_ERROR = "omitting directory '{0}'"
OVERWRITE_ERROR = "cannot overwrite non-directory '{0}' with directory '{1}'"

def copyRootObject(sourceFile,sourcePathSplit,destFile,destPathSplit,optDict,oneSource=False):
    """
    Initialize the recursive function 'copyRootObjectRecursive', written to be as unix-like as possible
    """
    retcode = 0
    isMultipleInput = not (oneSource and sourcePathSplit != [])
    recursiveOption = optDict["recursive"]
    # Multiple input and unexisting or non-directory destination
    # TARGET_ERROR
    if isMultipleInput and destPathSplit != [] \
        and not (isExisting(destFile,destPathSplit) \
        and isDirectory(destFile,destPathSplit)):
        logging.warning(TARGET_ERROR.format(destPathSplit[-1]))
        retcode += 1
    # Entire ROOT file or directory in input omitting "-r" option
    # OMITTING_FILE_ERROR or OMITTING_DIRECTORY_ERROR
    if not recursiveOption:
        if sourcePathSplit == []:
            logging.warning(OMITTING_FILE_ERROR.format( \
                sourceFile.GetName()))
            retcode += 1
        elif isDirectory(sourceFile,sourcePathSplit):
            logging.warning(OMITTING_DIRECTORY_ERROR.format( \
                sourcePathSplit[-1]))
            retcode += 1
    # Run copyRootObjectRecursive function with the wish
    # to follow the unix copy behaviour
    if sourcePathSplit == []:
        retcode += copyRootObjectRecursive(sourceFile,sourcePathSplit, \
            destFile,destPathSplit,optDict)
    else:
        setName = ""
        if not isMultipleInput and (destPathSplit != [] \
            and not isExisting(destFile,destPathSplit)):
            setName = destPathSplit[-1]
        objectName = sourcePathSplit[-1]
        if isDirectory(sourceFile,sourcePathSplit):
            if setName != "":
                createDirectory(destFile,destPathSplit[:-1]+[setName])
                retcode += copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                    destFile,destPathSplit[:-1]+[setName],optDict)
            elif isDirectory(destFile,destPathSplit):
                if not isExisting(destFile,destPathSplit+[objectName]):
                    createDirectory(destFile,destPathSplit+[objectName])
                if isDirectory(destFile,destPathSplit+[objectName]):
                    retcode += copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                        destFile,destPathSplit+[objectName],optDict)
                else:
                    logging.warning(OVERWRITE_ERROR.format( \
                        objectName,objectName))
                    retcode += 1
            else:
                logging.warning(OVERWRITE_ERROR.format( \
                    destPathSplit[-1],objectName))
                retcode += 1
        else:
            if setName != "":
                retcode += copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                    destFile,destPathSplit[:-1],optDict,setName)
            elif isDirectory(destFile,destPathSplit):
                retcode += copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                    destFile,destPathSplit,optDict)
            else:
                setName = destPathSplit[-1]
                retcode += copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                    destFile,destPathSplit[:-1],optDict,setName)
    return retcode

DELETE_ERROR = "object {0} was not existing, so it is not deleted"

def deleteObject(rootFile,pathSplit):
    """
    Delete the object 'pathSplit[-1]' from (rootFile,pathSplit[:-1])
    """
    retcode = changeDirectory(rootFile,pathSplit[:-1])
    if retcode == 0:
        fileName = pathSplit[-1]
        if isExisting(rootFile,pathSplit):
            ROOT.gDirectory.Delete(fileName+";*")
        else:
            logging.warning(DELETE_ERROR.format(fileName))
            retcode += 1
    return retcode

def copyRootObjectRecursive(sourceFile,sourcePathSplit,destFile,destPathSplit,optDict,setName=""):
    """
    Copy objects from a file or directory (sourceFile,sourcePathSplit)
    to an other file or directory (destFile,destPathSplit)
    - Has the will to be unix-like
    - that's a recursive function
    - Python adaptation of a root input/output tutorial :
      $ROOTSYS/tutorials/io/copyFiles.C
    """
    retcode = 0
    replaceOption = optDict["replace"]
    for key in getKeyList(sourceFile,sourcePathSplit):
        objectName = key.GetName()
        if isDirectoryKey(key):
            if not isExisting(destFile,destPathSplit+[objectName]):
                createDirectory(destFile,destPathSplit+[objectName])
            if isDirectory(destFile,destPathSplit+[objectName]):
                retcode +=copyRootObjectRecursive(sourceFile, \
                    sourcePathSplit+[objectName], \
                    destFile,destPathSplit+[objectName],optDict)
            else:
                logging.warning(OVERWRITE_ERROR.format( \
                    objectName,objectName))
                retcode += 1
        elif isTreeKey(key):
            T = key.GetMotherDir().Get(objectName+";"+str(key.GetCycle()))
            if replaceOption and isExisting(destFile,destPathSplit+[T.GetName()]):
                retcodeTemp = deleteObject(destFile,destPathSplit+[T.GetName()])
                if retcodeTemp:
                    retcode += retcodeTemp
                    continue
            changeDirectory(destFile,destPathSplit)
            newT = T.CloneTree(-1,"fast")
            if setName != "":
                newT.SetName(setName)
            newT.Write()
        else:
            obj = key.ReadObj()
            if replaceOption and isExisting(destFile,destPathSplit+[setName]):
                changeDirectory(destFile,destPathSplit)
                otherObj = getFromDirectory(setName)
                if not otherObj == obj:
                    retcodeTemp = deleteObject(destFile,destPathSplit+[setName])
                    if retcodeTemp:
                        retcode += retcodeTemp
                        continue
                    else:
                        obj.SetName(setName)
                        changeDirectory(destFile,destPathSplit)
                        obj.Write()
                else:
                    obj.SetName(setName)
                    changeDirectory(destFile,destPathSplit)
                    obj.Write()
            else:
                if setName != "":
                    obj.SetName(setName)
                changeDirectory(destFile,destPathSplit)
                obj.Write()
            obj.Delete()
    changeDirectory(destFile,destPathSplit)
    ROOT.gDirectory.SaveSelf(ROOT.kTRUE)
    return retcode

FILE_REMOVE_ERROR = "cannot remove '{0}': Is a ROOT file"
DIRECTORY_REMOVE_ERROR = "cannot remove '{0}': Is a directory"
ASK_FILE_REMOVE = "remove '{0}' ? (y/n) : "
ASK_OBJECT_REMOVE = "remove '{0}' from '{1}' ? (y/n) : "

def deleteRootObject(rootFile,pathSplit,optDict):
    """
    Remove the object (rootFile,pathSplit)
    -interactive : prompt before every removal
    -recursive : allow directory, and ROOT file, removal
    """
    retcode = 0
    if not optDict["recursive"] and isDirectory(rootFile,pathSplit):
        if pathSplit == []:
            logging.warning(FILE_REMOVE_ERROR.format(rootFile.GetName()))
            retcode += 1
        else:
            logging.warning(DIRECTORY_REMOVE_ERROR.format(pathSplit[-1]))
            retcode += 1
    else:
        if optDict['interactive']:
            if pathSplit != []:
                answer = raw_input(ASK_OBJECT_REMOVE \
                    .format("/".join(pathSplit),rootFile.GetName()))
            else:
                answer = raw_input(ASK_FILE_REMOVE \
                    .format(rootFile.GetName()))
            remove = answer.lower() == 'y'
        else:
            remove = True
        if remove:
            if pathSplit != []:
                retcode += deleteObject(rootFile,pathSplit)
            else:
                rootFile.Close()
                os.remove(rootFile.GetName())
    return retcode

# End of functions shared by roocp, roomv and roorm
##########

##########
# Help strings for ROOT command line tools

# Arguments
SOURCE_HELP = "path of the source."
SOURCES_HELP = "path of the source(s)."
DEST_HELP = "path of the destination."

# Options
COMPRESS_HELP = \
"""change the compression settings of the
destination file (if not already existing)."""
INTERACTIVE_HELP = "prompt before every removal."
RECREATE_HELP = "recreate the destination file."
RECURSIVE_HELP = "recurse inside directories"
REPLACE_HELP = "replace object if already existing"

# End of help strings
##########

#!/usr/bin/python

# ROOT command line tools module: cmdLineUtils
# Author: Julien Ripoche
# Mail: julien.ripoche@u-psud.fr
# Date: 13/08/15

"""Contain utils for ROOT command line tools"""

##
# The code of the these functions can be found here : http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
# Thanks J.F. Sebastian !!

from contextlib import contextmanager
import os
import sys

def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextmanager
def streamRedirected(actual_output=sys.stdout, to=os.devnull):
    stdout_fd = fileno(actual_output)
    # copy stdout_fd before it is overwritten
    #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        actual_output.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield actual_output # allow code to be run with the redirected stream
        finally:
            # restore actual_output to its previous value
            #NOTE: dup2 makes stdout_fd inheritable unconditionally
            actual_output.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

def stdoutRedirected():
     return streamRedirected(sys.stdout, os.devnull)

def stderrRedirected():
     return streamRedirected(sys.stderr, os.devnull)

# Not on all version of python...
# with stdoutRedirected(), stderrRedirected():
#     ...

# The end of streamRedirected function
##

# redirect output (escape characters during ROOT importation...)
with stdoutRedirected():
    import ROOT

import argparse
import glob
import os
import sys
import fnmatch
import logging

def changeDirectory(rootFile,pathSplit):
    """Change the current directory (ROOT.gDirectory)
    by the corresponding directory (rootFile,pathSplit)"""
    # This function has no protection because 'rootFile' should be
    # a real ROOT file and 'pathSplit' a real directory path.
    # If an error occurs in this function it means that the
    # arguments are defective.
    rootFile.cd()
    retcode = 0
    for directoryName in pathSplit:
        theDir = ROOT.gDirectory.Get(directoryName)
        if not theDir:
            logging.warning("Directory %s does not exist." %directoryName)
            retcode +=1
        else:
            theDir.cd()
    return retcode


def createDirectory(rootFile,pathSplit):
    """Add a directory named 'pathSplit[-1]'
    in (rootFile,pathSplit[:-1])"""
    # To be used with "not isExisting(rootFile,pathSplit)"
    changeDirectory(rootFile,pathSplit[:-1])
    ROOT.gDirectory.mkdir(pathSplit[-1])

def deleteObject(rootFile,pathSplit):
    """Delete the object 'pathSplit[-1]'
    from (rootFile,pathSplit[:-1])"""
    # To be used with "isExisting(rootFile,pathSplit)"
    changeDirectory(rootFile,pathSplit[:-1])
    ROOT.gDirectory.Delete(pathSplit[-1]+";*")

def isExisting(rootFile,pathSplit):
    """Return True if the object, corresponding to
    (rootFile,pathSplit), exits"""
    changeDirectory(rootFile,pathSplit[:-1])
    return ROOT.gDirectory.GetListOfKeys().Contains(pathSplit[-1])

def isDirectoryKey(key):
    """Return True if the object, corresponding to the key,
    inherits from TDirectory, False if not"""
    classname = key.GetClassName()
    cl = ROOT.gROOT.GetClass(classname)
    return cl.InheritsFrom(ROOT.TDirectory.Class())

def isTreeKey(key):
    """Return True if the object, corresponding to the key,
    inherits from TTree, False if not"""
    classname = key.GetClassName()
    cl = ROOT.gROOT.GetClass(classname)
    return cl.InheritsFrom(ROOT.TTree.Class())

def getKey(rootFile,pathSplit):
    """Get the key of the corresponding object
    (rootFile,pathSplit)"""
    changeDirectory(rootFile,pathSplit[:-1])
    return ROOT.gDirectory.GetKey(pathSplit[-1])

def isDirectory(rootFile,pathSplit):
    """Return True if the object, corresponding to (rootFile,pathSplit),
    inherits from TDirectory, False if not"""
    if pathSplit == []: return True # the object is the rootFile itself
    else: return isDirectoryKey(getKey(rootFile,pathSplit))

def isTree(rootFile,pathSplit):
    """Return True if the object, corresponding to (rootFile,pathSplit),
    inherits from TTree, False if not"""
    if pathSplit == []: return False # the object is the rootFile itself
    else: return isTreeKey(getKey(rootFile,pathSplit))

ROOT_FILE_ERROR = "'{0}' is not a ROOT file"

def zombieExclusion(rootFile):
    """Print an error message and exit
    if rootFile is a zombie"""
    if rootFile.IsZombie():
        logging.error(ROOT_FILE_ERROR.format(rootFile.GetName()))
        sys.exit()

def getKeyList(rootFile,pathSplit):
    """Get the list of key of the directory (rootFile,pathSplit),
    if (rootFile,pathSplit) is not a directory then get the key"""
    if isDirectory(rootFile,pathSplit):
        changeDirectory(rootFile,pathSplit)
        return ROOT.gDirectory.GetListOfKeys()
    else: return [getKey(rootFile,pathSplit)]

def keyClassSpliter(rootFile,pathSplitList):
    """Separate directories and other objects
    for rools and rooprint"""
    objList = []
    dirList = []
    for pathSplit in pathSplitList:
        if pathSplit == []: dirList.append(pathSplit)
        elif isDirectory(rootFile,pathSplit): dirList.append(pathSplit)
        else: objList.append(pathSplit)
    return objList,dirList

def asHashableList(pathSplitList):
    """Make hashable list of pathSplitList joining with '/'"""
    hashableList = []
    for n in pathSplitList:
        hashableList.append("/".join(n))
    return hashableList

MANY_OCCURENCE_WARNING = "'{0}' appears many times in '{1}'"

def manyOccurenceWarning(hashableList,fileName):
    """Search for double occurence of the same object name"""
    hashableList.sort()
    for i,n in enumerate(hashableList):
        if len(hashableList)!=1:
            if i != len(hashableList)-1:
                if hashableList[i+1] == n:
                    logging.warning(MANY_OCCURENCE_WARNING.format(n,fileName))
            else:
                if hashableList[0] == n:
                    logging.warning(MANY_OCCURENCE_WARNING.format(n,fileName))

def patternToPathSplitList(fileName,pattern):
    """Get the list of pathSplit of objects in the ROOT file
    corresponding to fileName that match with the pattern"""
    # avoid multiple slash problem
    patternSplit = [n for n in pattern.split("/") if n != ""]
    # whole ROOT file, so unnecessary to open it
    if patternSplit == []: return [[]]
    # redirect output (missing dictionary for class...)
    with stderrRedirected():
        rootFile = ROOT.TFile(fileName)
    zombieExclusion(rootFile)
    pathSplitList = [[]]
    for patternPiece in patternSplit:
        newPathSplitList = []
        for pathSplit in pathSplitList:
            ## Stay in top level of trees
            #if isTree(rootFile,pathSplit[:-1]):
            #    continue
            #elif isDirectory(rootFile,pathSplit):
            if isDirectory(rootFile,pathSplit):
                changeDirectory(rootFile,pathSplit)
                newPathSplitList.extend( \
                    [pathSplit + [key.GetName()] \
                    for key in ROOT.gDirectory.GetListOfKeys() \
                    if fnmatch.fnmatch(key.GetName(),patternPiece)])
            ## Equivalent for tree inspection
            #elif isTree(rootFile,pathSplit):
            #    changeDirectory(rootFile,pathSplit[:-1])
            #    T = ROOT.gDirectory.Get(pathSplit[-1])
            #    newPathSplitList.extend( \
            #        [pathSplit + [branch.GetName()] \
            #        for branch in T.GetListOfBranches() \
            #        if fnmatch.fnmatch(branch.GetName(),patternPiece)])
        pathSplitList = newPathSplitList
    if pathSplitList == []: # no match...
        logging.warning("can't find {0} in {1}".format(pattern,fileName))
    hashableList = asHashableList(pathSplitList)
    if len(set(hashableList)) != len(hashableList): # same match...
        manyOccurenceWarning(hashableList,fileName)
    return pathSplitList

def patternToFileNameAndPathSplitList(pattern,wildcards = True):
    """Get the list of tuple containing both :
    - ROOT file name
    - list of path splited (in the corresponding file)
      of object that matches
    Use unix wildcards by default"""
    fileList = []
    patternSplit = pattern.split(":")
    if patternSplit[0] in ["http","https","ftp"]: # file from the web
        patternSplit[0] += ":"+patternSplit[1]
        del patternSplit[1]
        fileNameList = [patternSplit[0]]
    else: fileNameList = \
        [os.path.expandvars(os.path.expanduser(i)) \
        for i in glob.iglob(patternSplit[0])] \
        if wildcards else \
        [os.path.expandvars(os.path.expanduser(patternSplit[0]))]
    #fileList = []
    #patternSplit = pattern.rsplit(":", 1)
    #fileNameList = \
    #    [os.path.expandvars(os.path.expanduser(i)) \
    #    for i in glob.iglob(patternSplit[0])] \
    #    if wildcards else \
    #    [os.path.expandvars(os.path.expanduser(patternSplit[0]))]
    #if fileNameList == []:
    #    fileNameList == [patternSplit[0]]
    for fileName in fileNameList:
        # there is a pattern of path in the ROOT file
        if len(patternSplit)>1 : pathSplitList = \
           patternToPathSplitList(fileName,patternSplit[1]) \
           if wildcards else \
           [[n for n in patternSplit[1].split("/") if n != ""]]
        else: pathSplitList = [[]] # whole ROOT file
        fileList.append((fileName,pathSplitList))
    return fileList

TARGET_ERROR = "target '{0}' is not a directory"
OMITTING_FILE_ERROR = "omitting file '{0}'"
OMITTING_DIRECTORY_ERROR = "omitting directory '{0}'"
OVERWRITE_ERROR = "cannot overwrite non-directory '{0}' with directory '{1}'"

def copyRootObject(sourceFile,sourcePathSplit,destFile,destPathSplit,optDict,oneSource=False):
    """Initialize the recursive function 'copyRootObjectRecursive',
    written to be as unix-like as possible"""
    isMultipleInput = not (oneSource and sourcePathSplit != [])
    recursiveOption = optDict["recursive"]
    # Multiple input and unexisting or non-directory destination
    # TARGET_ERROR
    if isMultipleInput and destPathSplit != [] \
        and not (isExisting(destFile,destPathSplit) \
        and isDirectory(destFile,destPathSplit)):
        logging.warning(TARGET_ERROR.format(destPathSplit[-1]))
        return
    # Entire ROOT file or directory in input omitting "-r" option
    # OMITTING_FILE_ERROR or OMITTING_DIRECTORY_ERROR
    if not recursiveOption:
        if sourcePathSplit == []:
            logging.warning(OMITTING_FILE_ERROR.format( \
                sourceFile.GetName()))
            return
        if isDirectory(sourceFile,sourcePathSplit):
            logging.warning(OMITTING_DIRECTORY_ERROR.format( \
                sourcePathSplit[-1]))
            return
    # Run copyRootObjectRecursive function with the wish
    # to follow the unix copy behaviour
    if sourcePathSplit == []:
        copyRootObjectRecursive(sourceFile,sourcePathSplit, \
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
                copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                    destFile,destPathSplit[:-1]+[setName],optDict)
            elif isDirectory(destFile,destPathSplit):
                if not isExisting(destFile,destPathSplit+[objectName]):
                    createDirectory(destFile,destPathSplit+[objectName])
                if isDirectory(destFile,destPathSplit+[objectName]):
                    copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                        destFile,destPathSplit+[objectName],optDict)
                else:
                    logging.warning(OVERWRITE_ERROR.format( \
                        objectName,objectName))
            else:
                logging.warning(OVERWRITE_ERROR.format( \
                    destPathSplit[-1],objectName))
        else:
            if isDirectory(destFile,destPathSplit):
                copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                    destFile,destPathSplit,optDict)
            else:
                copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                    destFile,destPathSplit[:-1],optDict,setName)

def copyRootObjectRecursive(sourceFile,sourcePathSplit,destFile,destPathSplit,optDict,setName=""):
    """Copy objects from a file or directory (sourceFile,sourcePathSplit)
    to an other file or directory (destFile,destPathSplit)
    - Has the will to be unix-like
    - that's a recursive function
    - Python adaptation of a root input/output tutorial :
      $ROOTSYS/tutorials/io/copyFiles.C"""
    replaceOption = optDict["replace"]
    for key in getKeyList(sourceFile,sourcePathSplit):
        objectName = key.GetName()
        if isDirectoryKey(key):
            if not isExisting(destFile,destPathSplit+[objectName]):
                createDirectory(destFile,destPathSplit+[objectName])
            if isDirectory(destFile,destPathSplit+[objectName]):
                copyRootObjectRecursive(sourceFile, \
                    sourcePathSplit+[objectName], \
                    destFile,destPathSplit+[objectName],optDict)
            else:
                logging.warning(OVERWRITE_ERROR.format( \
                    objectName,objectName))
        elif isTreeKey(key):

            T = key.GetMotherDir().Get(objectName+";"+str(key.GetCycle()))
            if replaceOption and isExisting(destFile,destPathSplit+[T.GetName()]):
                deleteObject(destFile,destPathSplit+[T.GetName()])
            changeDirectory(destFile,destPathSplit)
            newT = T.CloneTree(-1,"fast")
            if setName != "":
                newT.SetName(setName)
            newT.Write()
        else:
            obj = key.ReadObj()
            if replaceOption and isExisting(destFile,destPathSplit+[obj.GetName()]):
                deleteObject(destFile,destPathSplit+[obj.GetName()])
            if setName != "":
                obj.SetName(setName)
            changeDirectory(destFile,destPathSplit)
            obj.Write()
            obj.Delete()
    changeDirectory(destFile,destPathSplit)
    ROOT.gDirectory.SaveSelf(ROOT.kTRUE)

FILE_REMOVE_ERROR = "cannot remove '{0}': Is a ROOT file"
DIRECTORY_REMOVE_ERROR = "cannot remove '{0}': Is a directory"
ASK_FILE_REMOVE = "remove '{0}' ? (y/n) : "
ASK_OBJECT_REMOVE = "remove '{0}' from '{1}' ? (y/n) : "

def deleteRootObject(rootFile,pathSplit,optDict):
    """Remove the object (rootFile,pathSplit)
    -interactive : prompt before every removal
    -recursive : allow directory, and ROOT file, removal"""
    if not optDict["recursive"] and isDirectory(rootFile,pathSplit):
        if pathSplit == []:
            logging.warning(FILE_REMOVE_ERROR.format(rootFile.GetName()))
        else:
            logging.warning(DIRECTORY_REMOVE_ERROR.format(pathSplit[-1]))
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
                deleteObject(rootFile,pathSplit)
            else:
                rootFile.Close()
                os.remove(rootFile.GetName())

##
# Help strings for ROOT command line tools

# Arguments
SOURCE_HELP = \
    "path of the source."
SOURCES_HELP = \
    "path of the source(s)."
DEST_HELP = \
    "path of the destination."

# Options
COMPRESS_HELP = \
    "change the compression settings of the destination file" + \
    "(if not already existed)."
RECREATE_HELP = \
    "recreate the destination file."

RECURSIVE_HELP = "Recurse inside directories"

# End of help strings
##


def createDirectory(rootFile,pathSplit):
    """Add a directory named 'pathSplit[-1]' in (rootFile,pathSplit[:-1])"""
    retcode = changeDirectory(rootFile,pathSplit[:-1])
    if retcode == 0:
        ROOT.gDirectory.mkdir(pathSplit[-1])
    return retcode

def _getParser(theHelp, theEpilog):
   """
   Get a commandline parser with the defaults of the commandline utils.
   """
   return argparse.ArgumentParser(description=theHelp,
                                  formatter_class=argparse.RawDescriptionHelpFormatter,
                                  epilog = theEpilog)

def getParserFile(theHelp, theEpilog=""):
   """
   Get a commandline parser with the defaults of the commandline utils and a
   source list of files
   """
   parser = _getParser(theHelp, theEpilog)
   parser.add_argument("FILE", nargs='+', help="Input file")
   return parser

def getParserSourceDest(theHelp, theEpilog=""):
   """
   Get a commandline parser with the defaults of the commandline utils and a
   source list of files
   """
   parser = _getParser(theHelp, theEpilog)
   parser.add_argument("SOURCE", nargs='+', help="Source file")
   parser.add_argument("DEST", help="Destination file")
   return parser

def getSourceListArgs(parser, wildcards = True):
   """
   Create a list of tuples that contain source ROOT file names
   and lists of path in these files as well as a dictionary with options
   """

   args = parser.parse_args()

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
    Create a list of tuples that contain source ROOT file names
    and lists of path in these files as well as a dictionary with options
    """
    sourceList, args = getSourceListArgs(parser, wildcards)
    return sourceList, vars(args)

def getSourceDestListOptDict(parser, wildcards = True):
    """
    Create a list of tuples that contain source ROOT file names
    and lists of path in these files as well as a dictionary with options
    """
    sourceList, args = getSourceListArgs(parser, wildcards)

    # Create a tuple that contain a destination ROOT file name
    # and a path in this file
    destList = \
        patternToFileNameAndPathSplitList( \
        args.DEST,wildcards=False)
    destFileName,destPathSplitList = destList[0]
    destPathSplit = destPathSplitList[0]


    return sourceList, destFileName, destPathSplit, vars(args)

def openROOTFile(fileName, mode="read"):
    theFile = ROOT.TFile.Open(str(fileName), mode)
    if not theFile:
        logging.warning("File %s does not exist", fileName)
    return theFile

def getFromDirectory(name):
    return ROOT.gDirectory.Get(name)

def openBrowser(rootfile=None):
    if rootfile:
        rootFile.Browse(ROOT.TBrowser())
    ROOT.PyROOT.TPyROOTApplication.Run(ROOT.gApplication)

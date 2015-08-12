#!/usr/bin/python

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

#Not on all version of python...
#with stdoutRedirected(), stderrRedirected():
    #...

# The end of stdoutRedirected function
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
    for directoryName in pathSplit:
        ROOT.gDirectory.Get(directoryName).cd()

def getKey(rootFile,pathSplit):
    """Get the key of the corresponding object
    (rootFile,pathSplit)"""
    changeDirectory(rootFile,pathSplit[:-1])
    return ROOT.gDirectory.GetKey(pathSplit[-1])

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

def isDirectory(rootFile,pathSplit):
    """Return True if the object, corresponding to (rootFile,pathSplit),
    inherits from TDirectory, False if not"""
    if pathSplit == []: return True # the object is the rootFile itself
    else: return isDirectoryKey(getKey(rootFile,pathSplit))

def isTreeKey(key):
    """Return True if the object, corresponding to the key,
    inherits from TTree, False if not"""
    classname = key.GetClassName()
    cl = ROOT.gROOT.GetClass(classname)
    return cl.InheritsFrom(ROOT.TTree.Class())

def isTree(rootFile,pathSplit):
    """Return True if the object, corresponding to (rootFile,pathSplit),
    inherits from TTree, False if not"""
    if pathSplit == []: return False # the object is the rootFile itself
    else: return isTreeKey(getKey(rootFile,pathSplit))

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

def patternToPathSplitList(fileName,pattern):
    """Get the list of pathSplit of objects in the ROOT file
    corresponding to fileName that match with the pattern"""
    # avoid multiple slash problem
    patternSplit = [n for n in pattern.split("/") if n != ""]
    # whole ROOT file, so unnecessary to open it
    if patternSplit == []: return [[]]
    # redirect output (missing dictionary for class...)
    with stderrRedirected():
        rootFile = ROOT.TFile.Open(fileName)
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
    if pathSplitList == []: # no match
        logging.warning("Can't find {0} in {1}".format(pattern,fileName))
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
OVERWRITE_ERROR = "cannot overwrite non-directory '{0}' with directory '{1}'"

def copyRootObject(sourceFile,sourcePathSplit,destFile,destPathSplit,optDict,oneSource=False):
    """Initialize the recursive function 'copyRootObjectRecursive',
    written to be as unix-like as possible"""
    isMultipleInput = not (oneSource and sourcePathSplit != [])
    toContinue = True
    if isMultipleInput:
        if destPathSplit != [] and not (isExisting(destFile,destPathSplit) \
            and isDirectory(destFile,destPathSplit)):
            logging.warning(TARGET_ERROR.format(destPathSplit[-1]))
            toContinue = False
    if toContinue:
        setName = ""
        if not isMultipleInput and (destPathSplit != [] \
            and not isExisting(destFile,destPathSplit)):
            setName = destPathSplit[-1]
            del destPathSplit[-1]
        if sourcePathSplit != []:
            key = getKey(sourceFile,sourcePathSplit)
            if isDirectoryKey(key):
                if setName != "":
                    changeDirectory(destFile,destPathSplit)
                    ROOT.gDirectory.mkdir(setName)
                    copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                        destFile,destPathSplit+[setName],optDict)
                elif isDirectory(destFile,destPathSplit):
                    changeDirectory(destFile,destPathSplit)
                    if not ROOT.gDirectory.GetListOfKeys().Contains( \
                        key.GetName()):
                        ROOT.gDirectory.mkdir(key.GetName())
                        copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                            destFile,destPathSplit+[key.GetName()],optDict)
                    elif isDirectory(destFile,destPathSplit+[key.GetName()]):
                        copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                            destFile,destPathSplit+[key.GetName()],optDict)
                    else:
                        logging.warning(OVERWRITE_ERROR.format( \
                            destPathSplit[-1]+"/"+key.GetName(),key.GetName()))
                else:
                    logging.warning(OVERWRITE_ERROR.format( \
                        destPathSplit[-1],key.GetName()))
            else:
                if setName != "":  
                    copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                        destFile,destPathSplit,optDict,setName)
                elif isDirectory(destFile,destPathSplit):
                    copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                        destFile,destPathSplit,optDict)
                else:
                    copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                        destFile,destPathSplit[:-1],optDict)
        else:
            copyRootObjectRecursive(sourceFile,sourcePathSplit, \
                destFile,destPathSplit,optDict)

def copyRootObjectRecursive(sourceFile,sourcePathSplit,destFile,destPathSplit,optDict,setName=""):
    """Copy objects from a file or directory (sourceFile,sourcePathSplit)
    to an other file or directory (destFile,destPathSplit)
    - Has the will to be unix-like
    - that's a recursive function
    - Python adaptation of a root input/output tutorial :
      $ROOTSYS/tutorials/io/copyFiles.C"""
    for key in getKeyList(sourceFile,sourcePathSplit):
        if isDirectoryKey(key):
            changeDirectory(destFile,destPathSplit)
            ## copy T dir dans T tree
            if not ROOT.gDirectory.GetListOfKeys().Contains(key.GetName()):
                ROOT.gDirectory.mkdir(key.GetName())
                copyRootObjectRecursive(sourceFile, \
                    sourcePathSplit+[key.GetName()], \
                    destFile,destPathSplit+[key.GetName()],optDict)
            elif isDirectory(destFile,destPathSplit+[key.GetName()]):
                copyRootObjectRecursive(sourceFile, \
                    sourcePathSplit+[key.GetName()], \
                    destFile,destPathSplit+[key.GetName()],optDict)
            else:
                logging.warning(OVERWRITE_ERROR.format( \
                    destPathSplit[-1]+"/"+key.GetName(),key.GetName()))
        elif isTreeKey(key):
            T = key.GetMotherDir().Get(key.GetName()+";"+str(key.GetCycle()))
            changeDirectory(destFile,destPathSplit)
            if optDict["replace"] and ROOT.gDirectory \
                .GetListOfKeys().Contains(T.GetName()):
                ROOT.gDirectory.Delete(T.GetName()+";*")
            newT = T.CloneTree(-1,"fast")
            if setName != "":
                newT.SetName(setName)
            newT.Write()
        else:
            obj = key.ReadObj()
            changeDirectory(destFile,destPathSplit)
            if optDict["replace"] and ROOT.gDirectory \
                .GetListOfKeys().Contains(obj.GetName()):
                ROOT.gDirectory.Delete(obj.GetName()+";*")
            if setName != "":
                obj.SetName(setName)
            obj.Write()
            obj.Delete()
    changeDirectory(destFile,destPathSplit)
    ROOT.gDirectory.SaveSelf(ROOT.kTRUE)

def deleteRootObject(rootFile,pathSplit,optDict={'i':False}):
    """Remove the object (rootFile,pathSplit)
    -i prompt before every removal"""
    answer = 'y'
    if optDict['i']: answer = \
       raw_input("Are you sure to remove '{0}' from '{1}' ? (y/n) : " \
                 .format("/".join(pathSplit),rootFile.GetName())) \
       if pathSplit != [] else \
       raw_input("Are you sure to remove '{0}' ? (y/n) : " \
                 .format(rootFile.GetName()))
    if answer.lower() == 'y':
        if pathSplit != []:
            changeDirectory(rootFile,pathSplit[:-1])
            ROOT.gDirectory.Delete(pathSplit[-1]+";*")
        else:
            rootFile.Close()
            os.remove(rootFile.GetName())

# Help strings for ROOT command line tools
SOURCE_HELP = \
    "path of the source."
SOURCES_HELP = \
    "path of the source(s)."
DEST_HELP = \
    "path of the destination."
COMPRESS_HELP = \
    "change the compression settings of the destination file" + \
    "(if not already existed)."
RECREATE_HELP = \
    "recreate the destination file."

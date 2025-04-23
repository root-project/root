#! /usr/bin/env python
# Author: Danilo Piparo, July 2014
"""
This program allows to parse selection xmls to check their syntax.
It has three arguments: the chunk size, the chunk index and the directory of the xmls.
The chunk size is the number of selection xml files to be treated.
The chunk index is the index of the bunch of size chunk size to be treated.
For example, if 31 selection xml files in total are available:
1) size 1, index 2: will analyze the 3rd xml
2) size 10, index 0: will analyze in bulk the first 10 xmls
3) size 10, index 2: will analyze the 31st xml (the reminder)
This utility has been written to be used within the ctest infrastructure.
"""
from __future__ import print_function
import os
import subprocess
import sys

emptyHeaderName="emptyHeader_%s.h"

def touch(fname, times=None):
  with open(fname, 'a'):
    os.utime(fname, times)

def getXMLsList(XMLPath):
  return list(filter(lambda s: s.endswith(".xml"),os.listdir(XMLPath)))

def chunkXMLList(XMLList,chunkSize):
  return [XMLList[x:x+chunkSize] for x in range(0, len(XMLList), chunkSize)]

def executeGenreflex(XMLFileName,headerName):
  """
  Example command: 
  genreflex emptyHeader.h  --quiet -o dict.cxx -s experimentsSelectionXMLs/mySelection.xml
  """
  retcode=0
  dictName=os.path.basename(XMLFileName)+"dummy_dict.cxx"
  genreflexCommand="genreflex %s -o %s -s %s --selSyntaxOnly" %(headerName,dictName, XMLFileName)
  print("Parsing %s" %XMLFileName)
  try:
    subprocess.check_output(genreflexCommand, shell=True)
  except subprocess.CalledProcessError as inst:
   retcode=inst.returncode
  return retcode

def runTests(chunkSize,chunkIndex,XMLPath):
  if chunkIndex<0:
    print("Chunk index cannot be negative!")
    sys.exit(1)
  XMLList=getXMLsList(XMLPath)
  XMLChunks=chunkXMLList(XMLList,chunkSize)   
  if chunkIndex>=len(XMLChunks):
    print("Chunk index is %s while the chunks are %s" %(chunkIndex,len(XMLChunks)))
    sys.exit(1)  
  retcode=0
  print("Analysing chunk %s, which consist of %s files" %(chunkIndex,len(XMLChunks[chunkIndex])))
  headerName=emptyHeaderName %chunkIndex
  touch(headerName)
  for XMLFileName in XMLChunks[chunkIndex]:
    retcode+=executeGenreflex(os.path.join(XMLPath,XMLFileName),headerName)
  return retcode

if __name__ == "__main__":
  #FIXME: the dummy is present for the --fixcling imposed to python scripts by ctest
  if len(sys.argv) != 5:
    print("Usage: %s DUMMY ChunkSize ChunkIndex XMLpath" %os.path.basename(__file__))
    sys.exit(1)
  dummy, chunkSizes,chunkIndexs,XMLPath = sys.argv[1:]
  retcode = runTests(int(chunkSizes),int(chunkIndexs),XMLPath)
  sys.exit(retcode)

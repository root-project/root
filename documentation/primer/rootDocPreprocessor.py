#! /usr/bin/env python
"""
A tool to preprocess root documentation in order to provide missing
functionality in the documentation tools like doxygen or pandoc.
At the moment filters are just functions. A proper implementation would
go through the creation of a filter class which is very genereic and can be
composed with a trigger and an action.
"""


import sys
import os

usage="Usage: %s fileToBeTreated.md\n" %__file__

def printErrAndUsage(msg):
   print msg
   print usage

def checkInvocation(argv):
   """
   Check if the invocation is correct. If not print help
   """
   retcode=0
   if len(argv) != 2:
      printErrAndUsage("Number of arguments different from 2")
      retcode+=1
   if not os.path.exists(argv[1]):
      printErrAndUsage("File %s does not exist." %argv[1])
      retcode+=1
   return retcode

def createPreprocessedName(filename):
   """
   Create a name for the preprocessed version of filename
   """
   fileName, fileExtension = os.path.splitext(filename)
   return "%s_preprocessed%s" %(fileName, fileExtension)

def includeFilter(text):
   """
   If a line of the form
   @ROOT_INCLUDE_FILE filename
   is encountered, it is replaced with the content of filename
   """
   newText=""
   retcode=0
   for line in text.split("\n"):
      if line.startswith("@ROOT_INCLUDE_FILE"):
         inclFileName = line.split()[1]
         if not os.path.exists(inclFileName):
            print "[includeFilter] Error: file %s does not exist." %inclFileName
            retcode+=1
            continue
         for inclFileLine in open(inclFileName).readlines():
            newText+=inclFileLine
      else:
         newText+=line+"\n"
   return retcode, newText


textFilters=[includeFilter]

def applyFilters(filename):
   """
   Apply a series of filters to the file  modifiying its content.
   A new file is created with the suffix _preprocessed before the extension.
   For example:
   myChapter.md --> myChapter_preprocessed.md
   """
   filecontent = open(filename).read()
   retcode=0
   for textFilter in textFilters:
      tmpretcode,filecontent=textFilter(filecontent)
      retcode+=tmpretcode
   preprocessedName = createPreprocessedName(filename)
   ofile = open(preprocessedName,"w")
   ofile.write(filecontent)
   return retcode


def preprocessFile():
   """
   Apply a series of "filters" to the input file.
   """
   argv=sys.argv
   retcode = checkInvocation(argv)
   if 0 != retcode:
      return retcode

   filename = argv[1]
   retcode = applyFilters(filename)
   if retcode != 0:
      print "Errors during the preprocessing."
   return retcode

if __name__ == "__main__":
   sys.exit(preprocessFile())

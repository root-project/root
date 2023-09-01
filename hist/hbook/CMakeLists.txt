# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT hist/hbook package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Hbook
  HEADERS
    THbookBranch.h
    THbookFile.h
    THbookKey.h
    THbookTree.h
  SOURCES
    src/THbookBranch.cxx
    src/THbookFile.cxx
    src/THbookKey.cxx
    src/THbookTree.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    minicern
  DEPENDENCIES
    Hist
    Matrix
    Tree
    Graf
    TreePlayer
    RIO
)

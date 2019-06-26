# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT html package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Html
  HEADERS
    TClassDocOutput.h
    TDocDirective.h
    TDocInfo.h
    TDocOutput.h
    TDocParser.h
    THtml.h
  SOURCES
    src/TClassDocOutput.cxx
    src/TDocDirective.cxx
    src/TDocInfo.cxx
    src/TDocOutput.cxx
    src/TDocParser.cxx
    src/THtml.cxx
  DEPENDENCIES
    Graf
    Thread
)

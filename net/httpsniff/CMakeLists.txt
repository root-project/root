# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/http package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RHTTPSniff
  HEADERS
    TRootSnifferFull.h
  SOURCES
    src/TRootSnifferFull.cxx
  DEPENDENCIES
    Gpad
    Graf
    Hist
    RHTTP
    RIO
    Thread
    Tree
    XMLIO
)

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

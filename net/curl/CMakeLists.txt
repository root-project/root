# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/curl package
# @author Jakob Blomer, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RCurlHttp
  HEADERS
    ROOT/RCurlConnection.hxx
    ROOT/RRawFileCurl.hxx
    TCurlFile.h
  SOURCES
    src/RCurlConnection.cxx
    src/RRawFileCurl.cxx
    src/TCurlFile.cxx
  LIBRARIES
    CURL::libcurl
  DEPENDENCIES
    RIO
)

if(NOT MSVC)
  target_compile_options(RCurlHttp PRIVATE -Wno-deprecated-declarations)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)

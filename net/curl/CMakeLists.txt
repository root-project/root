# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/curl package
# @author Jakob Blomer, CERN
############################################################################

set(RCURLHTTP_HEADERS
  ROOT/RCurlConnection.hxx
  ROOT/RRawFileCurl.hxx
  TCurlFile.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(RCurlHttp
  HEADERS
    ${RCURLHTTP_HEADERS}
  SOURCES
    src/RCurlConnection.cxx
    src/RRawFileCurl.cxx
    src/TCurlFile.cxx
  LIBRARIES
    CURL::libcurl
  DEPENDENCIES
    RIO
)

set(RELATIVE_INC_HEADERS ${RCURLHTTP_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(RCurlHttp
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

if(NOT MSVC)
  target_compile_options(RCurlHttp PRIVATE -Wno-deprecated-declarations)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)

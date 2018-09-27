############################################################################
# CMakeLists.txt file for building ROOT net/davix package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RDAVIX
  HEADERS
    TDavixFile.h
    TDavixSystem.h
  SOURCES
    TDavixFile.cxx
    TDavixSystem.cxx
  LIBRARIES
    Davix::Davix
  DEPENDENCIES
    Net
    RIO
)

target_compile_options(RDAVIX PRIVATE -Wno-deprecated-declarations)

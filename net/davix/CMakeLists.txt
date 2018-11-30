############################################################################
# CMakeLists.txt file for building ROOT net/davix package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RDAVIX
  HEADERS
    TDavixFile.h
    TDavixSystem.h
  SOURCES
    src/TDavixFile.cxx
    src/TDavixSystem.cxx
  DEPENDENCIES
    Net
    RIO
)

target_compile_options(RDAVIX PRIVATE -Wno-deprecated-declarations)
target_link_libraries(RDAVIX PUBLIC Davix::Davix)

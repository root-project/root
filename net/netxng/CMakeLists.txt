############################################################################
# CMakeLists.txt file for building ROOT net/netxng package
# @author Lukasz Janyst <ljanyst@cern.ch>
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(NetxNG
  HEADERS
    TNetXNGFile.h
    TNetXNGFileStager.h
    TNetXNGSystem.h
  SOURCES
    src/TNetXNGFile.cxx
    src/TNetXNGFileStager.cxx
    src/TNetXNGSystem.cxx
  LIBRARIES
    ${XROOTD_LIBRARIES}
  DEPENDENCIES
    Net
    RIO
    Thread
  BUILTINS
    XROOTD
)

target_compile_options(NetxNG PRIVATE -Wno-shadow)
target_include_directories(NetxNG PUBLIC ${XROOTD_INCLUDE_DIRS})

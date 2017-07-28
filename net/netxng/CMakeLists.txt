############################################################################
# CMakeLists.txt file for building ROOT net/netxng package
# @author Lukasz Janyst <ljanyst@cern.ch>
############################################################################

include_directories(${XROOTD_INCLUDE_DIRS})
add_definitions(${XROOTD_CFLAGS})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-shadow)

ROOT_STANDARD_LIBRARY_PACKAGE(NetxNG
                              LIBRARIES ${XROOTD_LIBRARIES}
                              DEPENDENCIES Net RIO Thread
                              BUILTINS XROOTD)

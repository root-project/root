############################################################################
# CMakeLists.txt file for building ROOT net/netx package
# @author Pere Mato, CERN
############################################################################

include_directories(${XROOTD_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/proof/proofd/inc)
add_definitions(${XROOTD_CFLAGS})
#---------------------------------------------------------------------------

ROOT_STANDARD_LIBRARY_PACKAGE(Netx
                              LIBRARIES ${XROOTD_LIBRARIES}
                              DEPENDENCIES Net RIO Thread
                              BUILTINS XROOTD)

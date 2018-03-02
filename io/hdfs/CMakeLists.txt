############################################################################
# CMakeLists.txt file for building ROOT io/hdfs package
############################################################################

include_directories(${HDFS_INCLUDE_DIRS})

add_definitions(-D_FILE_OFFSET_BITS=64)

ROOT_STANDARD_LIBRARY_PACKAGE(HDFS
                              LIBRARIES ${HDFS_LIBRARIES}
                              DEPENDENCIES Net RIO Core)

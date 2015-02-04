# - Locate hdfs (from hadoop) library
# Defines:
#
#  HDFS_FOUND
#  HDFS_INCLUDE_DIR
#  HDFS_INCLUDE_DIRS (not cached)
#  HDFS_LIBRARIES

find_path(HDFS_INCLUDE_DIR NAMES hdfs.h  HINTS ${HDFS_DIR}/include $ENV{HDFS_DIR}/include)
find_library(HDFS_LIBRARY NAMES hdfs HINTS ${HDFS_DIR}/lib/native $ENV{HDFS_DIR}/lib/native)

set(HDFS_INCLUDE_DIRS ${HDFS_INCLUDE_DIR})
set(HDFS_LIBRARIES ${HDFS_LIBRARY})


# handle the QUIETLY and REQUIRED arguments and set HDFS_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(HDFS DEFAULT_MSG HDFS_INCLUDE_DIR HDFS_LIBRARY)

mark_as_advanced(HDFS_FOUND HDFS_INCLUDE_DIR HDFS_LIBRARY)

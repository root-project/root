############################################################################
# CMakeLists.txt file for building ROOT io/hdfs package
############################################################################

include_directories(${HDFS_INCLUDE_DIRS})

add_definitions(-D_FILE_OFFSET_BITS=64)

ROOT_GENERATE_DICTIONARY(G__HDFS *.h MODULE HDFS LINKDEF LinkDef.h DEPENDENCIES Net RIO)

ROOT_LINKER_LIBRARY(HDFS *.cxx G__HDFS.cxx LIBRARIES Core ${HDFS_LIBRARIES} DEPENDENCIES Net RIO)
ROOT_INSTALL_HEADERS()

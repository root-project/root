############################################################################
# CMakeLists.txt file for building ROOT io/gfal package
############################################################################

include_directories(${GFAL_INCLUDE_DIRS})

add_definitions(-D_FILE_OFFSET_BITS=64)

ROOT_STANDARD_LIBRARY_PACKAGE(GFAL
                              LIBRARIES ${GFAL_LIBRARIES}
                              DEPENDENCIES Core Net RIO)

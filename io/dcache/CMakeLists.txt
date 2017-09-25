############################################################################
# CMakeLists.txt file for building ROOT io/dcache package
############################################################################

include_directories(SYSTEM ${DCAP_INCLUDE_DIRS})

ROOT_STANDARD_LIBRARY_PACKAGE(DCache
                              LIBRARIES ${DCAP_LIBRARIES}
                              DEPENDENCIES Core Net RIO)

############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res)

# No dictionary here: no classes, no I/O.

ROOT_OBJECT_LIBRARY(Clib *.c)

ROOT_INSTALL_HEADERS()





############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

set(Clib_dict_headers strlcpy.h
                      snprintf.h PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res)

ROOT_OBJECT_LIBRARY(Clib *.c)

ROOT_INSTALL_HEADERS()





############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

set(Clib_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/strlcpy.h
                      ${CMAKE_CURRENT_SOURCE_DIR}/inc/snprintf.h CACHE INTERNAL "")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res)

ROOT_OBJECT_LIBRARY(Clib *.c)

ROOT_INSTALL_HEADERS()

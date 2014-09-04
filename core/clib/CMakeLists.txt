############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

set(Clib_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/strlcpy.h
                      ${CMAKE_CURRENT_SOURCE_DIR}/inc/snprintf.h PARENT_SCOPE)

ROOT_OBJECT_LIBRARY(Clib *.c)

ROOT_INSTALL_HEADERS()





############################################################################
# CMakeLists.txt file for building ROOT core/cont package
############################################################################

ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
set(Cont_dict_headers ${headers} PARENT_SCOPE)

ROOT_OBJECT_LIBRARY(Cont *.cxx)

ROOT_INSTALL_HEADERS()

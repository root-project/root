############################################################################
# CMakeLists.txt file for building ROOT core/winnt package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

set(Winnt_dict_headers  ${CMAKE_CURRENT_SOURCE_DIR}/inc/TWinNTSystem.h PARENT_SCOPE)

ROOT_OBJECT_LIBRARY(Winnt ${sources})

ROOT_INSTALL_HEADERS()

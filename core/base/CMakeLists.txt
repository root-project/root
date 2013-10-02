############################################################################
# CMakeLists.txt file for building ROOT core/base package
############################################################################

ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/T*.h)
set(Base_dict_headers ${headers} PARENT_SCOPE)

include_directories(${CMAKE_SOURCE_DIR}/graf3d/g3d/inc)
ROOT_OBJECT_LIBRARY(Base *.cxx)

if(builtin_pcre)
  add_dependencies(Base PCRE)
endif()

ROOT_INSTALL_HEADERS()

#---Extra options-----------------------------------------------------------------------
set_source_files_properties(src/TPRegexp.cxx COMPILE_FLAGS -I${PCRE_INCLUDE_DIR})
if(GCC_MAJOR EQUAL 4 AND GCC_MINOR EQUAL 1)
  set_source_files_properties(src/TString.cxx src/TContextMenu.cxx
                              COMPILE_FLAGS -Wno-strict-aliasing )
endif()


############################################################################
# CMakeLists.txt file for building ROOT core/base package
############################################################################

ROOT_GLOB_HEADERS(Base_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/T*.h)
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src src/*.cxx)

include_directories(${CMAKE_SOURCE_DIR}/graf3d/g3d/inc)
include_directories(${CMAKE_SOURCE_DIR}/gui/gui/inc)
include_directories(${CMAKE_SOURCE_DIR}/io/io/inc)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../textinput/src)
include_directories(${CMAKE_SOURCE_DIR})

ROOT_GENERATE_DICTIONARY(G__Core
                         ${Base_dict_headers}
                         ${Clib_dict_headers}
                         ${Cont_dict_headers}
                         ${Macosx_dict_headers}
                         ${Unix_dict_headers}
                         ${Winnt_dict_headers}
                         ${MetaUtils_dict_headers}
                         ${Meta_dict_headers}
                         ${TextInput_dict_headers}
                         STAGE1
                         MODULE Core
                         OPTIONS ${dict_opts}
                         LINKDEF LinkDef.h)

list(REMOVE_ITEM sources TROOT.cxx roota.cxx)
ROOT_OBJECT_LIBRARY(Base G__Core.cxx ${sources})

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


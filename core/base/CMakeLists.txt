############################################################################
# CMakeLists.txt file for building ROOT core/base package
############################################################################

ROOT_GLOB_HEADERS(Base_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/T*.h
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/GuiTypes.h 
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/KeySymbols.h 
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/Buttons.h)
if(root7)
    set(root7src v7/src/*.cxx)
    ROOT_GLOB_HEADERS(Base_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)
endif()
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src src/*.cxx ${root7src})

ROOT_GENERATE_DICTIONARY(G__Core
                         ${Base_dict_headers}
                         ${Base_v7_dict_headers}
                         ${Clib_dict_headers}
                         ${Cont_dict_headers}
                         ${Foundation_dict_headers}
                         ${Macosx_dict_headers}
                         ${Unix_dict_headers}
                         ${Winnt_dict_headers}
                         ${ClingUtils_dict_headers}
                         ${Meta_dict_headers}
                         ${TextInput_dict_headers}
                         STAGE1
                         MODULE Core
                         OPTIONS ${dict_opts}
                         LINKDEF LinkDef.h)

list(REMOVE_ITEM sources TROOT.cxx roota.cxx)
ROOT_OBJECT_LIBRARY(Base G__Core.cxx ${sources})

ROOT_INSTALL_HEADERS()

#---Extra options-----------------------------------------------------------------------
set_source_files_properties(src/TPRegexp.cxx COMPILE_FLAGS -I${PCRE_INCLUDE_DIR})
if(GCC_MAJOR EQUAL 4 AND GCC_MINOR EQUAL 1)
  set_source_files_properties(src/TString.cxx src/TContextMenu.cxx
                              COMPILE_FLAGS -Wno-strict-aliasing )
endif()


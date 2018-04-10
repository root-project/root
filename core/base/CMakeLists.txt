############################################################################
# CMakeLists.txt file for building ROOT core/base package
############################################################################

ROOT_ADD_TEST_SUBDIRECTORY(test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res)

ROOT_GLOB_HEADERS(Base_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef?.h
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/T*.h
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/Rtypes.h
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/GuiTypes.h
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/MessageTypes.h
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/KeySymbols.h
                                    ${CMAKE_CURRENT_SOURCE_DIR}/inc/Buttons.h
				    ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/TSequentialExecutor.hxx)
if(root7)
    set(root7src v7/src/*.cxx)
    ROOT_GLOB_HEADERS(Base_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/v7/inc/ROOT/T*.hxx)
endif()
ROOT_GLOB_SOURCES(sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/src src/*.cxx ${root7src})

# libCore is linked against zlib (based on the location returned by find_package or builtin_zlib). The dictionary of
# libCore, G__Core.cxx, should contain the include path to zlib.h in order to keep the symbol and header files in sync.
# The includePath variable of the dictionary is empty, but we do not notice it because, by chance, G__MathMore adds
# the same includePath for GSL. OTOH, if we have a small test case which works only with libCore.so we never give a
# chance MathMore's dictionary to hide the problem and thus we pick up the system zlib.h.
#
# FIXME: Using include_directories is too intrusive as it does much more than what we want, that is, it will add this
# include path not only in the dictionary but as part of the compilation lines. It is harmless but suboptimal: we need a
# mechanism throough which we can express that we only need the path in the dictionary.
#
# NOTE: We cannot use set_property(TARGET G__Core ...) becase this has to happen before registering the target.
include_directories(${ZLIB_INCLUDE_DIR})

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

# This is needed becuase LinkDef.h includes other LinkDef starting from  ${CMAKE_SOURCE_DIR}
set_property(TARGET G__Core PROPERTY INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR})

list(REMOVE_ITEM sources TROOT.cxx roota.cxx)
ROOT_OBJECT_LIBRARY(Base G__Core.cxx ${sources} BUILTINS PCRE)

ROOT_INSTALL_HEADERS()

#---Extra options-----------------------------------------------------------------------
set_source_files_properties(src/TPRegexp.cxx COMPILE_FLAGS -I${PCRE_INCLUDE_DIR})
if(GCC_MAJOR EQUAL 4 AND GCC_MINOR EQUAL 1)
  set_source_files_properties(src/TString.cxx src/TContextMenu.cxx
                              COMPILE_FLAGS -Wno-strict-aliasing )
endif()


# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT (global) core package
############################################################################

add_subdirectory(clib)
add_subdirectory(clingutils)
add_subdirectory(cont)
add_subdirectory(dictgen)
add_subdirectory(foundation)
add_subdirectory(meta)
add_subdirectory(metacling)
if(NOT WIN32)
add_subdirectory(multiproc)
endif()
add_subdirectory(rint)
add_subdirectory(textinput)
add_subdirectory(thread)
add_subdirectory(imt)
add_subdirectory(zip)
add_subdirectory(lzma)
add_subdirectory(lz4)

if(NOT WIN32)
  add_subdirectory(newdelete)
endif()

add_subdirectory(base)

if(UNIX)
  add_subdirectory(unix)
  set(unix_objects $<TARGET_OBJECTS:Unix>)
endif()
if(WIN32)
  add_subdirectory(winnt)
  set(winnt_objects $<TARGET_OBJECTS:Winnt>)
endif()
if(cocoa)
  add_subdirectory(macosx)
  set(macosx_objects $<TARGET_OBJECTS:Macosx>)
endif()

if(CMAKE_CXX_STANDARD GREATER 11)
  set(dict_v7dirs base/v7/inc)
endif()

#---G__Core--------------------------------------------------------------------

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
include_directories(
  ${ZLIB_INCLUDE_DIR}
  base/inc
  ${dict_v7dirs}
  clib/inc
  cont/inc
  foundation/inc
  macosx/inc
  unix/inc
  winnt/inc
  clingutils/inc
  meta/inc
  textinput/inc
)

set(objectlibs $<TARGET_OBJECTS:Base>
               $<TARGET_OBJECTS:Clib>
               $<TARGET_OBJECTS:Cont>
               $<TARGET_OBJECTS:Foundation>
               $<TARGET_OBJECTS:Lzma>
               $<TARGET_OBJECTS:Lz4>
               $<TARGET_OBJECTS:Zip>
               $<TARGET_OBJECTS:Meta>
               $<TARGET_OBJECTS:TextInput>
               ${macosx_objects}
               ${unix_objects}
               ${winnt_objects})

#---Generation of RGitCommit.h-----------------------------------------------------------
foreach(exp ${objectlibs})
  string(REGEX REPLACE "^[$]<TARGET_OBJECTS:(.+)>" "\\1" lib ${exp})
  get_target_property(objs ${lib} OBJECTS)
  list(APPEND dep_objects ${objs})
  list(APPEND dep_targets ${lib})
endforeach()

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/include/RGitCommit.h
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp ${CMAKE_BINARY_DIR}/include/RGitCommit.h
                   COMMENT ""
                   DEPENDS ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
                   WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

message("Recording the git revision now")

file(WRITE ${CMAKE_BINARY_DIR}/etc/gitinfo.txt
     "${GIT_DESCRIBE_ALL}
${GIT_DESCRIBE_ALWAYS}
${GIT_TIMESTAMP}
"
    )

file(WRITE ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
     "#ifndef ROOT_RGITCOMMIT_H
#define ROOT_RGITCOMMIT_H
#define ROOT_GIT_BRANCH \"${GIT_DESCRIBE_ALL}\"
#define ROOT_GIT_COMMIT \"${GIT_DESCRIBE_ALWAYS}\"
#endif
"
    )

add_custom_target(gitcommit ALL DEPENDS ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp)
set_source_files_properties(${CMAKE_BINARY_DIR}/RGitCommit.h.tmp PROPERTIES GENERATED TRUE)
add_dependencies(gitcommit ${dep_targets})

set_source_files_properties(${CMAKE_BINARY_DIR}/include/RGitCommit.h
                            PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)

ROOT_OBJECT_LIBRARY(BaseTROOT ${CMAKE_SOURCE_DIR}/core/base/src/TROOT.cxx ${CMAKE_BINARY_DIR}/include/RGitCommit.h)
add_dependencies(BaseTROOT gitcommit)
install(FILES ${CMAKE_BINARY_DIR}/include/RGitCommit.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
#----------------------------------------------------------------------------------------

if(WIN32)
  set(corelinklibs shell32.lib WSock32.lib Oleaut32.lib Iphlpapi.lib)
elseif(APPLE)
  if(cocoa)
     set(corelinklibs "-framework Cocoa -F/System/Library/PrivateFrameworks -framework CoreSymbolication")
  else()
     set(corelinklibs "-F/System/Library/PrivateFrameworks -framework CoreSymbolication")
  endif()
endif()

add_subdirectory(rootcling_stage1)

#-------------------------------------------------------------------------------
ROOT_LINKER_LIBRARY(Core $<TARGET_OBJECTS:BaseTROOT> ${objectlibs} BUILTINS LZMA)

ROOT_GENERATE_DICTIONARY(G__Core
  ${Core_dict_headers}
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
  NOTARGET
  MODULE
    Core
  OPTIONS
    -writeEmptyRootPCM
    ${dict_opts}
  LINKDEF
    base/inc/LinkDef.h
)

# This is needed because LinkDef.h includes other LinkDef starting from ${CMAKE_SOURCE_DIR}
target_include_directories(Core PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(Core
  PRIVATE
    PCRE::PCRE
    ${LIBLZMA_LIBRARIES}
    xxHash::xxHash
    LZ4::LZ4
    ZLIB::ZLIB
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${corelinklibs}
)

add_dependencies(Core CLING)

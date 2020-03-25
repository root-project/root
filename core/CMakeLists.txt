# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT (global) core package
############################################################################

file(WRITE ${CMAKE_BINARY_DIR}/etc/gitinfo.txt
  "${GIT_DESCRIBE_ALL}\n${GIT_DESCRIBE_ALWAYS}\n${GIT_TIMESTAMP}\n")

file(WRITE ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
"#ifndef ROOT_RGITCOMMIT_H
#define ROOT_RGITCOMMIT_H
  #define ROOT_GIT_BRANCH \"${GIT_DESCRIBE_ALL}\"
  #define ROOT_GIT_COMMIT \"${GIT_DESCRIBE_ALWAYS}\"
#endif"
)

if(ROOT_ARCHITECTURE MATCHES macosx)
  list(APPEND platform_incl core/unix/inc)
  list(APPEND platform_incl core/macosx/inc)
elseif(ROOT_ARCHITECTURE MATCHES win32)
  set(platform_incl core/winnt/inc)
else()
  set(platform_incl core/unix/inc)
endif()

ROOT_CONFIGURE_LIBRARY_INCLUDES(Core
   GDIRS
      ${CMAKE_BINARY_DIR}/ginclude
   DIRS
      core/base/inc
      core/foundation/inc
      core/cont/inc
      core/gui/inc
      core/meta/inc
      core/clib/inc
      core/zip/inc
      core/thread/inc
      core/textinput/inc
      ${platform_incl}
   V7DIRS
      core/base/v7/inc
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/ginclude/RGitCommit.h
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
    ${CMAKE_BINARY_DIR}/ginclude/RGitCommit.h
  DEPENDS
    ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
)

add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/include/RGitCommit.h
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
    ${CMAKE_BINARY_DIR}/include/RGitCommit.h
  DEPENDS
    ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
)

add_custom_target(gitcommit ALL DEPENDS ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp ${CMAKE_BINARY_DIR}/include/RGitCommit.h ${CMAKE_BINARY_DIR}/ginclude/RGitCommit.h)

set_source_files_properties(${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
  PROPERTIES GENERATED TRUE)

set_source_files_properties(${CMAKE_BINARY_DIR}/ginclude/RGitCommit.h
  PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)

set_source_files_properties(${CMAKE_BINARY_DIR}/include/RGitCommit.h
  PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)

install(FILES ${CMAKE_BINARY_DIR}/include/RGitCommit.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_subdirectory(clib)
add_subdirectory(clingutils)
add_subdirectory(cont)
add_subdirectory(dictgen)
add_subdirectory(foundation)
add_subdirectory(gui)
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
add_subdirectory(zstd)
if(asan)
  add_subdirectory(sanitizer)
endif()

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

# Now includes path used from Core module, which configured before

set(objectlibs $<TARGET_OBJECTS:Base>
               $<TARGET_OBJECTS:Clib>
               $<TARGET_OBJECTS:Cont>
               $<TARGET_OBJECTS:Foundation>
               $<TARGET_OBJECTS:Lzma>
               $<TARGET_OBJECTS:Lz4>
               $<TARGET_OBJECTS:Zstd>
               $<TARGET_OBJECTS:Zip>
               $<TARGET_OBJECTS:Meta>
               $<TARGET_OBJECTS:GuiCore>
               $<TARGET_OBJECTS:TextInput>
               ${macosx_objects}
               ${unix_objects}
               ${winnt_objects})

ROOT_OBJECT_LIBRARY(BaseTROOT ${CMAKE_SOURCE_DIR}/core/base/src/TROOT.cxx ${CMAKE_BINARY_DIR}/ginclude/RGitCommit.h INCLUDES foundation/res)
add_dependencies(BaseTROOT gitcommit)
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

if (libcxx AND NOT APPLE)
  # In case we use libcxx and glibc together there is a mismatch of the
  # signatures of functions in the header wchar.h. This macro tweaks the
  # header in rootcling resource directory to be compatible with the one from
  # libc++.
  target_compile_definitions(Core PRIVATE __CORRECT_ISO_CPP_WCHAR_H_PROTO)
endif()


ROOT_GENERATE_DICTIONARY(G__Core
  ${Core_dict_headers}
  ${Clib_dict_headers}
  ${Cont_dict_headers}
  ${Foundation_dict_headers}
  ${Macosx_dict_headers}
  ${Unix_dict_headers}
  ${Winnt_dict_headers}
  ${ClingUtils_dict_headers}
  ${GuiCore_dict_headers}
  ${Meta_dict_headers}
  ${TextInput_dict_headers}
  STAGE1
  MODULE
    Core
  OPTIONS
    -writeEmptyRootPCM
  LINKDEF
    base/inc/LinkDef.h
)

# This is needed because LinkDef.h includes other LinkDef starting from ${CMAKE_SOURCE_DIR}/core
target_include_directories(G__Core PRIVATE ${CMAKE_SOURCE_DIR}/core)
target_include_directories(G__Core PRIVATE ${CMAKE_SOURCE_DIR}/core/clingutils/inc)


target_link_libraries(Core
  PRIVATE
    PCRE::PCRE
    ${LIBLZMA_LIBRARIES}
    xxHash::xxHash
    LZ4::LZ4
    ZLIB::ZLIB
    ${ZSTD_LIBRARIES}
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${corelinklibs}
)

add_dependencies(Core CLING)

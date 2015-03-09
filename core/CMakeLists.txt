############################################################################
# CMakeLists.txt file for building ROOT (global) core package
############################################################################

ROOT_USE_PACKAGE(core/base)
ROOT_USE_PACKAGE(core/clib)
ROOT_USE_PACKAGE(core/cont)
ROOT_USE_PACKAGE(core/meta)
ROOT_USE_PACKAGE(core/metautils)
ROOT_USE_PACKAGE(core/textinput)
ROOT_USE_PACKAGE(core/thread)
ROOT_USE_PACKAGE(core/unix)
ROOT_USE_PACKAGE(core/winnt)
ROOT_USE_PACKAGE(core/macosx)
ROOT_USE_PACKAGE(core/zip)
ROOT_USE_PACKAGE(core/lzma)

if(builtin_pcre)
  add_subdirectory(pcre)
endif()
add_subdirectory(rint)
add_subdirectory(thread)
if(NOT WIN32)
  add_subdirectory(newdelete)
endif()
add_subdirectory(clib)
add_subdirectory(cont)
add_subdirectory(textinput)
add_subdirectory(metautils)
add_subdirectory(meta)
set(systemdict_opts)
if(UNIX)
  add_subdirectory(unix)
  set(unix_objects $<TARGET_OBJECTS:Unix>)
  set(dict_opts -DSYSTEM_TYPE_unix ${dict_opts})
endif()
if(WIN32)
  add_subdirectory(winnt)
  set(winnt_objects $<TARGET_OBJECTS:Winnt>)
  set(dict_opts -DSYSTEM_TYPE_winnt ${dict_opts})
endif()
if(cocoa)
  add_subdirectory(macosx)
  set(macosx_objects $<TARGET_OBJECTS:Macosx>)
  set(dict_opts -DSYSTEM_TYPE_macosx ${dict_opts})
endif()
add_subdirectory(zip)
add_subdirectory(lzma)
add_subdirectory(base)

set(objectlibs $<TARGET_OBJECTS:Base>
               $<TARGET_OBJECTS:Clib>
               $<TARGET_OBJECTS:Cont>
               $<TARGET_OBJECTS:Lzma>
               $<TARGET_OBJECTS:Zip>
               $<TARGET_OBJECTS:MetaUtils>
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
if(WIN32)
  add_custom_command(OUTPUT  ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
                     COMMAND ${CMAKE_SOURCE_DIR}/build/win/gitinfo.bat ${CMAKE_SOURCE_DIR}
                     COMMAND ${CMAKE_SOURCE_DIR}/build/win/githeader.bat RGitCommit.h.tmp
                     COMMENT "Recording the git revision now"
                     DEPENDS ${dep_objects}
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else()
  add_custom_command(OUTPUT  ${CMAKE_BINARY_DIR}/RGitCommit.h.tmp
                     COMMAND ${CMAKE_SOURCE_DIR}/build/unix/gitinfo.sh ${CMAKE_SOURCE_DIR}
                     COMMAND ${CMAKE_SOURCE_DIR}/build/unix/githeader.sh RGitCommit.h.tmp
                     COMMENT "Recording the git revision now"
                     DEPENDS ${dep_objects}
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
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
elseif(cocoa)
   set(corelinklibs "-framework Cocoa")
endif()
add_subdirectory(utils)

#-------------------------------------------------------------------------------
ROOT_LINKER_LIBRARY(Core
                    $<TARGET_OBJECTS:BaseTROOT>
                    ${objectlibs}
                    LIBRARIES ${PCRE_LIBRARIES} ${LZMA_LIBRARIES} ${ZLIB_LIBRARY}
                              ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${corelinklibs} )

if(cling)
  add_dependencies(Core CLING)
endif()
if(builtin_pcre)
  add_dependencies(Core PCRE)
endif()
if(builtin_lzma)
  ROOT_ADD_BUILTIN_DEPENDENCIES(Core LZMA)
endif()

#----------------------------------------------------------------------------------------

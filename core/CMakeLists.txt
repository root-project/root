############################################################################
# CMakeLists.txt file for building ROOT (global) core package
############################################################################

ROOT_USE_PACKAGE(core/base)
ROOT_USE_PACKAGE(core/clib)
ROOT_USE_PACKAGE(core/cont)
ROOT_USE_PACKAGE(core/meta)
ROOT_USE_PACKAGE(core/metautils)
ROOT_USE_PACKAGE(core/textinput)
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
add_subdirectory(newdelete)
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

if(WIN32)
   set(corelinklibs shell32.lib WSock32.lib Oleaut32.lib Iphlpapi.lib)
elseif(cocoa)
   set(corelinklibs "-framework Cocoa")
endif()
add_subdirectory(utils)

#-------------------------------------------------------------------------------


ROOT_LINKER_LIBRARY(Core
                    $<TARGET_OBJECTS:Base>
                    $<TARGET_OBJECTS:Clib>
                    $<TARGET_OBJECTS:Cont>
                    $<TARGET_OBJECTS:Lzma>
                    $<TARGET_OBJECTS:Zip>
                    $<TARGET_OBJECTS:MetaUtils>
                    $<TARGET_OBJECTS:Meta>
                    $<TARGET_OBJECTS:TextInput>
                    ${macosx_objects}
                    ${unix_objects}
                    ${winnt_objects}
                    LIBRARIES ${PCRE_LIBRARIES} ${LZMA_LIBRARIES} ${ZLIB_LIBRARY} 
                              ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${corelinklibs} )

add_dependencies(Core TEXTINPUT_DICTIONARY)

if(cling)
  add_dependencies(Core CLING)
endif()

Set(LibCore_SRCS)
Set(LibCore_INCLUDE_DIRS)
Set(LibCore_LINKDEF)

ROOT_USE_PACKAGE(core/base)
ROOT_USE_PACKAGE(core/clib)
ROOT_USE_PACKAGE(core/cont)
ROOT_USE_PACKAGE(core/meta)
ROOT_USE_PACKAGE(core/textinput)
ROOT_USE_PACKAGE(core/thread)
ROOT_USE_PACKAGE(core/unix)
ROOT_USE_PACKAGE(core/winnt)
ROOT_USE_PACKAGE(core/macosx)
ROOT_USE_PACKAGE(core/zip)
ROOT_USE_PACKAGE(core/lz4)
ROOT_USE_PACKAGE(core/lzma)
ROOT_USE_PACKAGE(cint/cint)


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
add_subdirectory(meta)
if(UNIX)
  add_subdirectory(unix)
endif()
if(WIN32)
  add_subdirectory(winnt)
endif()
if(cocoa)
  add_subdirectory(macosx)
endif()
add_subdirectory(zip)
add_subdirectory(lzma)
add_subdirectory(lz4)
add_subdirectory(base)
add_subdirectory(metautils)
add_subdirectory(utils)

set(CORE_DICTIONARIES
    ${METAUTILS_DICTIONARY}
    ${CLIB_DICTIONARY} 
    ${CONT_DICTIONARY} 
    ${TEXTINPUT_DICTIONARY} 
    ${META_DICTIONARY} 
    ${UNIX_DICTIONARY} 
    ${WINNT_DICTIONARY} 
    ${MACOSX_DICTIONARY} 
    ${BASE1_DICTIONARY} 
    ${BASE2_DICTIONARY} 
    ${BASE3_DICTIONARY} 
    ${BASE4_DICTIONARY} 
   ) 

set_source_files_properties(${METAUTILS_DICTIONARY} GENERATED)
set_source_files_properties(${CLIB_DICTIONARY} GENERATED)
set_source_files_properties(${CONT_DICTIONARY} GENERATED)
set_source_files_properties(${TEXTINPUT_DICTIONARY} GENERATED)
set_source_files_properties(${META_DICTIONARY} GENERATED)
if(UNIX)
  set_source_files_properties(${UNIX_DICTIONARY} GENERATED)
endif()
if(WIN32)
  set_source_files_properties(${WINNT_DICTIONARY} GENERATED)
endif()
if(cocoa)
  set_source_files_properties(${MACOSX_DICTIONARY} GENERATED)
endif()
set_source_files_properties(${BASE1_DICTIONARY} GENERATED)
set_source_files_properties(${BASE2_DICTIONARY} GENERATED)
set_source_files_properties(${BASE3_DICTIONARY} GENERATED)
set_source_files_properties(${BASE4_DICTIONARY} GENERATED)

set_source_files_properties(${TEXTINPUT_CXX_SOURCES} PROPERTIES
                             COMPILE_FLAGS -I${CMAKE_SOURCE_DIR}/core/textinput/src )


set_source_files_properties(${CMAKE_SOURCE_DIR}/core/base/src/TPRegexp.cxx
                            ${BASE1_DICTIONARY} ${BASE2_DICTIONARY}
                            COMPILE_FLAGS -I${PCRE_INCLUDE_DIR}
                           )
set_source_files_properties(${CMAKE_SOURCE_DIR}/core/lzma/src/ZipLZMA.c
                            COMPILE_FLAGS -I${LZMA_INCLUDE_DIR}
                           )

if(${GCC_MAJOR} EQUAL 4 AND ${GCC_MINOR} EQUAL 1)
  set_source_files_properties(${CMAKE_SOURCE_DIR}/core/base/src/TString.cxx
                              ${CMAKE_SOURCE_DIR}/core/base/src/TContextMenu.cxx
                              COMPILE_FLAGS -Wno-strict-aliasing
                             )
endif()

if(${ROOT_ARCHITECTURE} MATCHES linuxicc)
  set_source_files_properties(${BASE3_DICTIONARY} 
                              COMPILE_FLAGS -wd191)
endif()


set(INCLUDE_DIRECTORIES 
    ${CMAKE_SOURCE_DIR}/core/metautils/inc 
    ${CMAKE_SOURCE_DIR}/core/meta/inc 
    ${CMAKE_SOURCE_DIR}/core/cont/inc 
    ${CMAKE_SOURCE_DIR}/core/unix/inc 
    ${CMAKE_SOURCE_DIR}/graf3d/g3d/inc
    ${CMAKE_SOURCE_DIR}/net/net/inc
    ${CMAKE_SOURCE_DIR}/core/base/src 
    ${ROOT_INCLUDE_DIR} ) 

ROOT_GENERATE_ROOTMAP(Core LINKDEF ${LibCore_LINKDEF})

include_directories( ${INCLUDE_DIRECTORIES})

if(WIN32)
   set(corelinklibs shell32.lib WSock32.lib Oleaut32.lib Iphlpapi.lib)
elseif(cocoa)
   set(corelinklibs "-framework Cocoa")
endif()

#---Add flags to compile ObjC++
foreach(src ${LibCore_SRCS})
  if(${src} MATCHES "\\.mm$")
    set_source_files_properties(${src} COMPILE_FLAGS "-ObjC++")
  endif()
endforeach()

if(CXX_HAS_Wno-strict-aliasing)
  set_source_files_properties(${CORE_DICTIONARIES} PROPERTIES COMPILE_FLAGS -Wno-strict-aliasing)
endif()

ROOT_LINKER_LIBRARY(Core ${LibCore_SRCS} ${CORE_DICTIONARIES}
                    LIBRARIES ${PCRE_LIBRARIES} ${LZMA_LIBRARIES}  ${LZ4_LIBRARIES}  ${ZLIB_LIB} ${CMAKE_DL_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${corelinklibs}
                    DEPENDENCIES Cint)
add_Dependencies(Core CLIB_DICTIONARY CONT_DICTIONARY  META_DICTIONARY METAUTILS_DICTIONARY BASE_DICTIONARY)
if(UNIX)
  add_dependencies(Core UNIX_DICTIONARY)
endif()
if(WIN32)
  add_dependencies(Core WINNT_DICTIONARY)
endif()
if(cocoa)
  add_dependencies(Core MACOSX_DICTIONARY)
endif()
add_dependencies(Core TEXTINPUT_DICTIONARY)
if(builtin_pcre)
  add_dependencies(Core PCRE)
endif()
if(builtin_lzma)
  ROOT_ADD_BUILTIN_DEPENDENCIES(Core LZMA)
endif()
target_compile_definitions(Core PRIVATE ${LZ4_DEFINITIONS})
if(builtin_lz4)
  ROOT_ADD_BUILTIN_DEPENDENCIES(Core LZ4)
endif()


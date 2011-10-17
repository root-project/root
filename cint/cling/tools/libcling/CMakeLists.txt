set(LLVM_USED_LIBS
  clingUserInterface
  clingMetaProcessor
  clingInterpreter
  clingUITextInput
  clangFrontend
  clangSerialization
  clangDriver
  clangCodeGen
  clangParse
  clangSema
  clangAnalysis
  clangRewrite
  clangAST
  clangLex
  clangBasic
  )

#set( LLVM_LINK_COMPONENTS
#  ${LLVM_TARGETS_TO_BUILD}
#  asmparser
#  bitreader
#  bitwriter
#  codegen
#  support
#  mc
#  executionengine
#  target
#  jit
#  x86info
#  x86asmprinter
#)

set(SOURCES
  Dummy.cpp)

if( LLVM_ENABLE_PIC )
  set(SHARED_LIBRARY TRUE)
  add_cling_library(libcling ${SOURCES})

  set_target_properties(libcling
    PROPERTIES
    OUTPUT_NAME "libcling")

  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(LIBCLING_LINK_FLAGS
      "-Wl,-compatibility_version -Wl,1 -Wl,-dead_strip -Wl,-seg1addr -Wl,0xE0000000")
    set_target_properties(libcling
      PROPERTIES
      LINK_FLAGS "${LIBCLING_LINK_FLAGS}"
      INSTALL_NAME_DIR "@executable_path/../lib")
  endif()

  if(MSVC)
    # windows.h doesn't compile with /Za
    get_target_property(NON_ANSI_COMPILE_FLAGS libcling COMPILE_FLAGS)
    string(REPLACE "/Za" "" NON_ANSI_COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
    set_target_properties(libcling PROPERTIES
      COMPILE_FLAGS ${NON_ANSI_COMPILE_FLAGS})
  endif()

  set(LIBCLING_STATIC_TARGET_NAME libcling_static)
else()
  set(LIBCLING_STATIC_TARGET_NAME libcling)
endif()

if( NOT BUILD_SHARED_LIBS AND NOT WIN32 )
  add_cling_library(${LIBCLING_STATIC_TARGET_NAME} STATIC ${SOURCES})

  set_target_properties(${LIBCLING_STATIC_TARGET_NAME}
    PROPERTIES
    OUTPUT_NAME "libcling")
endif()

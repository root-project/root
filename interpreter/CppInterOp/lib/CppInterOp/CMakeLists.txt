if(EMSCRIPTEN)
  set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
  set(LLVM_LINK_COMPONENTS "")
else()
  set(LLVM_LINK_COMPONENTS
    ${LLVM_TARGETS_TO_BUILD}
    BinaryFormat
    Core
    Object
    OrcJit
    Support
  )
  # FIXME: Investigate why this needs to be conditionally included.
  if ("LLVMFrontendDriver" IN_LIST LLVM_AVAILABLE_LIBS)
    list(APPEND LLVM_LINK_COMPONENTS  FrontendDriver)
  endif()
  if ("LLVMOrcDebugging" IN_LIST LLVM_AVAILABLE_LIBS)
    list(APPEND LLVM_LINK_COMPONENTS OrcDebugging)
  endif()
endif()
  set(DLM
    DynamicLibraryManager.cpp
    DynamicLibraryManagerSymbol.cpp
    Paths.cpp
  )
  if (CPPINTEROP_USE_CLING)
    set(LLVM_OPTIONAL_SOURCES ${LLVM_OPTIONAL_SOURCES} ${DLM})
    set(DLM)
  endif(CPPINTEROP_USE_CLING)
  if (CPPINTEROP_USE_REPL)
    #Use DML optional sources
  endif(CPPINTEROP_USE_REPL)

  if (CPPINTEROP_USE_CLING)
    set(cling_clang_interp clingInterpreter)
  endif()
  if (CPPINTEROP_USE_REPL)
    set(cling_clang_interp clangInterpreter)
  endif()

if(EMSCRIPTEN)
  set(link_libs
    ${cling_clang_interp}
    )
else()
  set(link_libs
    ${cling_clang_interp}
    clangAST
    clangBasic
    clangFrontend
    clangLex
    clangSema
    )
endif()

  if(NOT WIN32 AND NOT EMSCRIPTEN)
    list(APPEND link_libs dl)
  endif()

  # Get rid of libLLVM-X.so which is appended to the list of static libraries.
  if (LLVM_LINK_LLVM_DYLIB)
    set(new_libs ${link_libs})
    set(libs ${new_libs})
    while(NOT "${new_libs}" STREQUAL "")
      foreach(lib ${new_libs})
        if(TARGET ${lib})
          get_target_property(transitive_libs ${lib} INTERFACE_LINK_LIBRARIES)
          if (NOT transitive_libs)
            continue()
          endif()
          foreach(transitive_lib ${transitive_libs})
            get_target_property(lib_type ${transitive_lib} TYPE)
            if("${lib_type}" STREQUAL "STATIC_LIBRARY")
              list(APPEND static_transitive_libs ${transitive_lib})
            else()
              # Filter our libLLVM.so and friends.
              continue()
            endif()
            if(NOT ${transitive_lib} IN_LIST libs)
              list(APPEND newer_libs ${transitive_lib})
              list(APPEND libs ${transitive_lib})
            endif()
          endforeach(transitive_lib)
          # Update the target properties with the list of only static libraries.
          set_target_properties(${lib} PROPERTIES INTERFACE_LINK_LIBRARIES "${static_transitive_libs}")
          set(static_transitive_libs "")
        endif()
      endforeach(lib)
      set(new_libs ${newer_libs})
      set(newer_libs "")
    endwhile()
    # We just got rid of the libLLVM.so and other components shipped as shared
    # libraries, we need to make up for the missing dependency.
    list(APPEND LLVM_LINK_COMPONENTS
      Coverage
      FrontendHLSL
      LTO
      )
    # We will need to append the missing dependencies to pull in the right
    # LLVM library dependencies.
    list(APPEND link_libs
      clangCodeGen
      clangStaticAnalyzerCore
      )
  endif(LLVM_LINK_LLVM_DYLIB)
  add_llvm_library(clangCppInterOp BUILDTREE_ONLY
    DISABLE_LLVM_LINK_LLVM_DYLIB
    CppInterOp.cpp
    CXCppInterOp.cpp
    ${DLM}
    LINK_LIBS
    ${link_libs}
  )


if(EMSCRIPTEN)
  if(BUILD_SHARED_LIBS)
  # FIXME: When dynamically linking the Emscripten shared library to the
  # unit tests main_module you get errors due to undefined symbols. The reading of the file
  # below into a SYMBOLS_LIST variable is a temporary workaround that exports the undefined
  # symbols from the shared library, until it can be determined why they are not being exported already.
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/exports.ld" SYMBOLS_LIST)

  # Replace newlines with spaces
  string(REPLACE "\n" " " SYMBOLS_LIST "${SYMBOLS_LIST}")

  set_target_properties(clangCppInterOp
    PROPERTIES NO_SONAME 1
  )
  target_link_options(clangCppInterOp
    PRIVATE "SHELL: -s WASM_BIGINT"
    PRIVATE "SHELL: -s SIDE_MODULE=1"
    PRIVATE "SHELL: ${SYMBOLS_LIST}"
  )
else()
  target_link_options(clangCppInterOp
    PRIVATE "SHELL: -s WASM_BIGINT"
  )
endif()
  if (CPPINTEROP_ENABLE_TESTING)
    # When compiling Emscripten tests the CppInterOp library it links to is expected to be in the same folder as the compiled Javascript
    add_custom_command(TARGET clangCppInterOp POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:clangCppInterOp> ${CMAKE_BINARY_DIR}/unittests/CppInterOp/
    )
  endif(CPPINTEROP_ENABLE_TESTING)

endif()

target_compile_definitions(clangCppInterOp PUBLIC "_CINDEX_LIB_") # workaround for the use of `CINDEX_LINKAGE`

string(REPLACE ";" "\;" _VER CPPINTEROP_VERSION)
set_source_files_properties(CppInterOp.cpp PROPERTIES COMPILE_DEFINITIONS
  "LLVM_BINARY_DIR=\"${LLVM_BINARY_DIR}\";CPPINTEROP_VERSION=\"${_VAR}\""
)

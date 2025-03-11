if(EMSCRIPTEN)
  set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)
  set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-s SIDE_MODULE=1")
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "-s SIDE_MODULE=1")
  set(CMAKE_STRIP FALSE)

  add_llvm_library(clangCppInterOp
    SHARED

    CppInterOp.cpp
    CXCppInterOp.cpp
    DynamicLibraryManager.cpp
    DynamicLibraryManagerSymbol.cpp
    Paths.cpp

    # Additional libraries from Clang and LLD
    LINK_LIBS
    clangInterpreter
  )
  #FIXME: Setting no_soname=1 is needed until https://github.com/emscripten-core/emscripten/blob/ac676d5e437525d15df5fd46bc2c208ec6d376a3/cmake/Modules/Platform/Emscripten.cmake#L36
  # is patched out of emsdk, as --soname is not recognised by emscripten. A PR to do this has been done here https://github.com/emscripten-core/emscripten/pull/23453 
  #FIXME: A patch is needed to llvm to remove -Wl,-z,defs since it is now recognised on emscripten. What needs to be removed is here
  # https://github.com/llvm/llvm-project/blob/128e2e446e90c3b1827cfc7d4d19e3c0976beff3/llvm/cmake/modules/HandleLLVMOptions.cmake#L318 . The PR to do try to do this is here 
  # https://github.com/llvm/llvm-project/pull/123396
  set_target_properties(clangCppInterOp
  PROPERTIES NO_SONAME 1
)
  target_link_options(clangCppInterOp PRIVATE
    PUBLIC "SHELL: -s WASM_BIGINT"
  )
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

  set(link_libs
    ${cling_clang_interp}
    clangAST
    clangBasic
    clangFrontend
    clangLex
    clangSema
    )

  if(NOT WIN32)
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

  add_llvm_library(clangCppInterOp
    DISABLE_LLVM_LINK_LLVM_DYLIB
    CppInterOp.cpp
    CXCppInterOp.cpp
    ${DLM}
    LINK_LIBS
    ${link_libs}
  )

  target_compile_definitions(clangCppInterOp PUBLIC "_CINDEX_LIB_") # workaround for the use of `CINDEX_LINKAGE`
endif()

string(REPLACE ";" "\;" _VER CPPINTEROP_VERSION)
set_source_files_properties(CppInterOp.cpp PROPERTIES COMPILE_DEFINITIONS
  "LLVM_BINARY_DIR=\"${LLVM_BINARY_DIR}\";CPPINTEROP_VERSION=\"${_VAR}\""
)

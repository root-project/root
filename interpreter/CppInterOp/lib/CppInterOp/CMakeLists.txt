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

# Set sources based on whether Cling or Clang-REPL is used
if (CPPINTEROP_USE_CLING)
  set(LLVM_OPTIONAL_SOURCES ${LLVM_OPTIONAL_SOURCES} ${DLM})
  set(DLM)
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
  # These make up for symbols the stripped libLLVM.so used to provide. Seed them
  # before the walk below, not after: the walk only drops the shared LLVM
  # dependency from the targets it visits, and clangStaticAnalyzerCore is not
  # reachable from the other link_libs -- appending it later leaves libLLVM.so
  # in DT_NEEDED.
  list(APPEND link_libs
    clangCodeGen
    clangStaticAnalyzerCore
    )
  set(new_libs ${link_libs})
  set(libs ${new_libs})
  # LLVM component libraries behind libLLVM.so that static clang archives
  # reference without declaring them in their link interfaces (the dylib
  # umbrella covered them). Substituted for the LLVM shared target below.
  # Do NOT expand to every LLVM library: some (e.g. LLVMInterpreter) require
  # optional dependencies (libffi) that may not be installed.
  set(llvm_component_libs LLVMFrontendDriver)
  foreach(component_lib ${llvm_component_libs})
    if(NOT TARGET ${component_lib})
      list(REMOVE_ITEM llvm_component_libs ${component_lib})
    endif()
  endforeach()
  while(NOT "${new_libs}" STREQUAL "")
    foreach(lib ${new_libs})
      if(TARGET ${lib})
        get_target_property(transitive_libs ${lib} INTERFACE_LINK_LIBRARIES)
        if (NOT transitive_libs)
          continue()
        endif()
        foreach(transitive_lib ${transitive_libs})
          # Replace the LLVM shared library (libLLVM.so) that packages like
          # Debian's declare as the link interface of the static clang archives
          # with the static LLVM component libraries: some LLVM libs are only
          # reachable through that umbrella (e.g. LLVMFrontendDriver for
          # llvm::driver::createTLII in clangCodeGen), so dropping it outright
          # produces unresolved symbols at link time.
          # Everything else must be kept verbatim: plain items (rt, dl, m) and
          # non-LLVM targets (e.g. ZLIB::ZLIB) are LLVM's system dependencies
          # that the static archives still need.
          if(TARGET ${transitive_lib}
             AND "${transitive_lib}" MATCHES "^LLVM")
            get_target_property(lib_type ${transitive_lib} TYPE)
            if("${lib_type}" STREQUAL "SHARED_LIBRARY"
               OR "${lib_type}" STREQUAL "MODULE_LIBRARY")
              foreach(component_lib ${llvm_component_libs})
                if(TARGET ${component_lib} AND NOT ${component_lib} IN_LIST static_transitive_libs)
                  list(APPEND static_transitive_libs ${component_lib})
                endif()
              endforeach()
              continue()
            endif()
          endif()
          list(APPEND static_transitive_libs ${transitive_lib})
          if(TARGET ${transitive_lib} AND NOT ${transitive_lib} IN_LIST libs)
            list(APPEND newer_libs ${transitive_lib})
            list(APPEND libs ${transitive_lib})
          endif()
        endforeach(transitive_lib)
        # Update the target properties with the filtered dependency list
        # (shared LLVM umbrella replaced by static component libraries).
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
    Option
    TargetParser
    )
endif(LLVM_LINK_LLVM_DYLIB)

# Generate .inc files from CppInterOp.td using cppinterop-tblgen.
# The .td source lives here; generated output goes to the build tree's
# include/CppInterOp/ so that CppInterOp.h can #include them.
#
# For cross-compilation (Emscripten), set CPPINTEROP_TABLEGEN_EXE to a
# pre-built native cppinterop-tblgen binary, or build one in a native
# build directory first.
set(CPPINTEROP_TD_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CPPINTEROP_TD_FILE ${CPPINTEROP_TD_DIR}/CppInterOp.td)
set(CPPINTEROP_TD_DEPS ${CPPINTEROP_TD_FILE}
                        ${CPPINTEROP_TD_DIR}/CppInterOpAPI.td)

if(CPPINTEROP_TABLEGEN_EXE)
  # Cross-compilation: use the provided native binary.
  set(CPPINTEROP_TBLGEN_CMD ${CPPINTEROP_TABLEGEN_EXE})
  set(CPPINTEROP_TBLGEN_DEPS ${CPPINTEROP_TD_DEPS})
elseif(NOT EMSCRIPTEN)
  # Native build: use the just-built target.
  set(CPPINTEROP_TBLGEN_CMD $<TARGET_FILE:cppinterop-tblgen>)
  set(CPPINTEROP_TBLGEN_DEPS cppinterop-tblgen ${CPPINTEROP_TD_DEPS})
endif()

if(CPPINTEROP_TBLGEN_CMD)
  file(MAKE_DIRECTORY ${CPPINTEROP_GEN_DIR})

  add_custom_command(
    OUTPUT ${CPPINTEROP_GEN_DIR}/CppInterOpAPI.inc
    COMMAND ${CPPINTEROP_TBLGEN_CMD}
      -I${CPPINTEROP_TD_DIR} -gen-cppinterop-api
      ${CPPINTEROP_TD_FILE} -o ${CPPINTEROP_GEN_DIR}/CppInterOpAPI.inc
    DEPENDS ${CPPINTEROP_TBLGEN_DEPS}
    COMMENT "Generating CppInterOpAPI.inc...")

  add_custom_command(
    OUTPUT ${CPPINTEROP_GEN_DIR}/CppInterOpDecl.inc
    COMMAND ${CPPINTEROP_TBLGEN_CMD}
      -I${CPPINTEROP_TD_DIR} -gen-cppinterop-decl
      ${CPPINTEROP_TD_FILE} -o ${CPPINTEROP_GEN_DIR}/CppInterOpDecl.inc
    DEPENDS ${CPPINTEROP_TBLGEN_DEPS}
    COMMENT "Generating CppInterOpDecl.inc...")

  add_custom_command(
    OUTPUT ${CPPINTEROP_GEN_DIR}/CXCppInterOpDecl.inc
    COMMAND ${CPPINTEROP_TBLGEN_CMD}
      -I${CPPINTEROP_TD_DIR} -gen-cx-cppinterop-decl
      ${CPPINTEROP_TD_FILE} -o ${CPPINTEROP_GEN_DIR}/CXCppInterOpDecl.inc
    DEPENDS ${CPPINTEROP_TBLGEN_DEPS}
    COMMENT "Generating CXCppInterOpDecl.inc...")

  add_custom_command(
    OUTPUT ${CPPINTEROP_GEN_DIR}/CXCppInterOpImpl.inc
    COMMAND ${CPPINTEROP_TBLGEN_CMD}
      -I${CPPINTEROP_TD_DIR} -gen-cx-cppinterop-impl
      ${CPPINTEROP_TD_FILE} -o ${CPPINTEROP_GEN_DIR}/CXCppInterOpImpl.inc
    DEPENDS ${CPPINTEROP_TBLGEN_DEPS}
    COMMENT "Generating CXCppInterOpImpl.inc...")

  add_custom_target(CppInterOpTableGen DEPENDS
    ${CPPINTEROP_GEN_DIR}/CppInterOpAPI.inc
    ${CPPINTEROP_GEN_DIR}/CppInterOpDecl.inc
    ${CPPINTEROP_GEN_DIR}/CXCppInterOpDecl.inc
    ${CPPINTEROP_GEN_DIR}/CXCppInterOpImpl.inc)
else()
  message(FATAL_ERROR
    "Cannot generate .inc files: set CPPINTEROP_TABLEGEN_EXE to a native "
    "cppinterop-tblgen binary when cross-compiling.")
endif()

# Filter cache entries to those CMake marks with HELPSTRING="No help,
# variable specified on the command line." -- the property is left on
# -D vars the project never reclaimed via set(... CACHE), so what
# survives is the user's extra args.
# See https://gitlab.kitware.com/cmake/cmake/-/work_items/19622
get_cmake_property(_cppinterop_cache_vars CACHE_VARIABLES)
set(CPPINTEROP_CMAKE_INVOCATION "")
foreach(_v ${_cppinterop_cache_vars})
  get_property(_help CACHE ${_v} PROPERTY HELPSTRING)
  if(NOT _help STREQUAL
     "No help, variable specified on the command line.")
    continue()
  endif()
  get_property(_type CACHE ${_v} PROPERTY TYPE)
  if(_type STREQUAL "UNINITIALIZED")
    set(_type "")
  else()
    set(_type ":${_type}")
  endif()
  string(APPEND CPPINTEROP_CMAKE_INVOCATION
         " -D${_v}${_type}=\"${${_v}}\"")
endforeach()
string(STRIP "${CPPINTEROP_CMAKE_INVOCATION}" CPPINTEROP_CMAKE_INVOCATION)
configure_file(BuildInfo.inc.in
               ${CPPINTEROP_GEN_DIR}/BuildInfo.inc
               @ONLY)

# If an embedding project opts out of CppInterOp's installs rules, route the
# library through LLVM's BUILDTREE_ONLY path so add_llvm_library() does not
# register it in LLVMExports or call install(TARGETS) for it.
set(_cppinterop_lib_install_args)
if(NOT CPPINTEROP_INSTALL)
  list(APPEND _cppinterop_lib_install_args BUILDTREE_ONLY)
endif()
add_llvm_library(clangCppInterOp
  ${_cppinterop_lib_install_args}
  DISABLE_LLVM_LINK_LLVM_DYLIB
  CppInterOp.cpp
  CppInterOpDispatch.cpp
  CXCppInterOp.cpp
  CXCppInterOpGenerated.cpp
  ErrorInternal.cpp
  Tracing.cpp
  ${DLM}
  LINK_LIBS
  ${link_libs}
)

# llvm_add_library() stamps LLVM's VERSION/SOVERSION on shared libraries on
# ELF platforms; unsetting them ships one unversioned file, as on Darwin.
if(NOT CPPINTEROP_SHARED_LIBRARY_VERSIONING)
  set_property(TARGET clangCppInterOp PROPERTY VERSION)
  set_property(TARGET clangCppInterOp PROPERTY SOVERSION)
endif()

if(EMSCRIPTEN)
  if(BUILD_SHARED_LIBS)
  # FIXME: When dynamically linking the Emscripten shared library to the
  # unit tests main_module you get errors due to undefined symbols. The reading of the file
  # below into a SYMBOLS_LIST variable is a temporary workaround that exports the undefined
  # symbols from the shared library, until it can be determined why they are not being exported already.
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/exports.ld" SYMBOLS_LIST)

  # Replace newlines with spaces
  string(REPLACE "\n" " " SYMBOLS_LIST "${SYMBOLS_LIST}")

  target_compile_options(clangCppInterOp
    PRIVATE "SHELL: -Oz"
    PRIVATE "SHELL: -flto"
    PRIVATE "SHELL: ${CPPINTEROP_EXTRA_WASM_FLAGS}"
  )
  target_link_options(clangCppInterOp
    PRIVATE "SHELL: -s WASM_BIGINT"
    PRIVATE "SHELL: -s SIDE_MODULE=1"
    PRIVATE "SHELL: ${SYMBOLS_LIST}"
    PRIVATE "SHELL: -Oz"
    PRIVATE "SHELL: -flto"
    PRIVATE "SHELL: ${CPPINTEROP_EXTRA_WASM_FLAGS}"
  )
  else()
  target_compile_options(clangCppInterOp
    PRIVATE "SHELL: ${CPPINTEROP_EXTRA_WASM_FLAGS}"
  )
  target_link_options(clangCppInterOp
      PRIVATE "SHELL: -s WASM_BIGINT"
      PRIVATE "SHELL: ${CPPINTEROP_EXTRA_WASM_FLAGS}"
  )
  endif(BUILD_SHARED_LIBS)
  if (CPPINTEROP_ENABLE_TESTING)
    # When compiling Emscripten tests the CppInterOp library it links to is expected to be in the same folder as the compiled Javascript
    add_custom_command(TARGET clangCppInterOp POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:clangCppInterOp> ${CMAKE_BINARY_DIR}/unittests/CppInterOp/
    )
  endif(CPPINTEROP_ENABLE_TESTING)
endif(EMSCRIPTEN)

if(TARGET CppInterOpTableGen)
  add_dependencies(clangCppInterOp CppInterOpTableGen)
endif()
target_compile_definitions(clangCppInterOp PUBLIC "_CINDEX_LIB_") # workaround for the use of `CINDEX_LINKAGE`
# The library defines its own dispatch trace-hook slots (Tracing.cpp): mark
# them dllexport on Windows via the CPPINTEROP_DISPATCH_STORAGE declarations.
target_compile_definitions(clangCppInterOp PRIVATE CPPINTEROP_DISPATCH_EXPORTS)

# Where clang keeps its builtin headers. MakeResourcesPath() derives this
# from LLVM_BINARY_DIR at runtime, which misses wherever clang's headers are
# installed under a different prefix than LLVM's libraries, leaving a probe
# for a `clang` on PATH as the only fallback. Ask the clang this build links
# against instead, while the answer is still knowable. The cache entry is
# the override; the detected value stays local so it cannot outlive a
# reconfigure against a different Clang_DIR.
set(CPPINTEROP_CLANG_RESOURCE_DIR "" CACHE PATH
  "Clang resource directory compiled into the library; detected when empty")

set(_clang_resource_dir "${CPPINTEROP_CLANG_RESOURCE_DIR}")
# Auto-detection is skipped for coverage-instrumented builds: a baked-in
# directory makes MakeResourcesPath() return early at runtime, leaving the
# runtime fallback paths below it uncovered. An explicit
# CPPINTEROP_CLANG_RESOURCE_DIR override is always honored.
if(NOT _clang_resource_dir AND NOT CODE_COVERAGE
   AND EXISTS "${CLANG_INSTALL_PREFIX}/bin/clang")
  execute_process(
    COMMAND "${CLANG_INSTALL_PREFIX}/bin/clang" -print-resource-dir
    OUTPUT_VARIABLE _detected_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(EXISTS "${_detected_resource_dir}/include")
    set(_clang_resource_dir "${_detected_resource_dir}")
  endif()
endif()
message(STATUS "Clang resource directory: ${_clang_resource_dir}")

string(REPLACE ";" "\;" _VER CPPINTEROP_VERSION)
set_source_files_properties(CppInterOp.cpp PROPERTIES COMPILE_DEFINITIONS
  "LLVM_BINARY_DIR=\"${LLVM_BINARY_DIR}\";CPPINTEROP_VERSION=\"${_VAR}\""
)

# The values land in C string literals, so escape for exactly that layer.
# They are delivered through a generated header (see BakedArgs.h.in) instead
# of -D definitions: with -D, the shell would apply another unescape layer
# and corrupt literal backslashes on the way to the preprocessor.
function(_baked_string_define OUT NAME VALUE)
  string(REPLACE "\\" "\\\\" _v "${VALUE}")
  string(REPLACE "\"" "\\\"" _v "${_v}")
  set(${OUT} "${${OUT}}#define ${NAME} \"${_v}\"\n" PARENT_SCOPE)
endfunction()
set(_baked_defines "")
if(CPPINTEROP_BAKED_INTERPRETER_ARGS)
  _baked_string_define(_baked_defines CPPINTEROP_BAKED_INTERPRETER_ARGS
    "${CPPINTEROP_BAKED_INTERPRETER_ARGS}")
endif()
if(_clang_resource_dir)
  _baked_string_define(_baked_defines CPPINTEROP_CLANG_RESOURCE_DIR
    "${_clang_resource_dir}")
endif()
configure_file(BakedArgs.h.in ${CMAKE_CURRENT_BINARY_DIR}/BakedArgs.h @ONLY)


#---Define the way we want to build and what of llvm/clang/cling------------------------------------

set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF)
set(LLVM_BUILD_TOOLS OFF)
set(LLVM_TARGETS_TO_BUILD X86 CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
set(LLVM_FORCE_USE_OLD_TOOLCHAIN ON)
# We only use llvm/clang through TCling which is (with the help of core/meta) already taking a lock
# to serialize access to llvm.  We can later review how to make this finer grained by using llvm's own locking
# mechanism.
set(LLVM_ENABLE_THREADS OFF)

#---Build type--------------------------------------------------------------------------------------
if(NOT DEFINED LLVM_BUILD_TYPE)
  set(LLVM_BUILD_TYPE Release CACHE STRING "Build type for LLVM (used to set CMAKE_BUILD_TYPE)")
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(LLVM_ENABLE_ASSERTIONS "YES")
else()
  if(LLVM_BUILD_TYPE STREQUAL "Debug")
    set(LLVM_ENABLE_ASSERTIONS "YES")
  else()
    set(LLVM_ENABLE_ASSERTIONS "NO")
  endif()
endif()
set(CMAKE_BUILD_TYPE ${LLVM_BUILD_TYPE})

#---Add the sub-directory exclusing all the targets from all-----------------------------------------
add_subdirectory(llvm/src EXCLUDE_FROM_ALL)

#---Mark the LLVM/CLANG variables as advanced--------------------------------------------------------
get_cmake_property(variables CACHE_VARIABLES)
foreach(var ${variables})
  if(var MATCHES "^(CLANG|LLVM|CLING)_")
    mark_as_advanced(FORCE ${var})
  endif()
endforeach()
mark_as_advanced(FORCE BUG_REPORT_URL BUILD_CLANG_FORMAT_VS_PLUGIN BUILD_SHARED_LIBS BUILD_TESTING 
                       C_INCLUDE_DIRS DEFAULT_SYSROOT FFI_INCLUDE_DIR FFI_LIBRARY_DIR
                       GCC_INSTALL_PREFIX LIBCLANG_BUILD_STATIC TOOL_INFO_PLIST)
mark_as_advanced(CLEAR LLVM_ENABLE_ASSERTIONS LLVM_BUILD_TYPE)
#---Set into parent scope LLVM_VERSION --------------------------------------------------------------
file(READ ${CMAKE_SOURCE_DIR}/interpreter/llvm/src/configure _filestr)
string(REGEX REPLACE ".*PACKAGE_VERSION='([0-9]+[.][0-9]+[.][0-9]+).*" "\\1" _llvm_version ${_filestr})
set(LLVM_VERSION ${_llvm_version} PARENT_SCOPE)

#---Added some missing dependencies------------------------------------------------------------------
add_dependencies(clingUtils clangCodeGen)
add_dependencies(clingInterpreter clangBasic clangAST ClangDriverOptions)
add_dependencies(clingMetaProcessor clangAST ClangDriverOptions)

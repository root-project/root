
#---Define the way we want to build and what of llvm/clang/cling------------------------------------

set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF)
set(LLVM_BUILD_TOOLS OFF)
set(LLVM_TARGETS_TO_BUILD X86 CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")
set(LLVM_FORCE_USE_OLD_TOOLCHAIN ON)

#---Build type--------------------------------------------------------------------------------------
if(NOT DEFINED LLVM_BUILD_TYPE)
  set(LLVM_BUILD_TYPE Release)
endif()
if(ROOT_BUILD_TYPE STREQUAL "Debug")
  set(LLVM_ENABLE_ASSERTIONS "YES")
else()
  if(LLVM_BUILD_TYPE STREQUAL "Debug")
    set(LLVM_ENABLE_ASSERTIONS "YES")
  else()
    set(LLVM_ENABLE_ASSERTIONS "NO")
  endif()
endif()
set(CMAKE_BUILD_TYPE ${LLVM_BUILD_TYPE})

#---Create a softlink of cling inside the tools of llvm----------------------------------------------
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                ${CMAKE_SOURCE_DIR}/interpreter/cling ${CMAKE_SOURCE_DIR}/interpreter/llvm/src/tools/cling)

#---Add the sub-directory exclusing all the targets from all-----------------------------------------
add_subdirectory(llvm/src EXCLUDE_FROM_ALL)

#---Added some missing dependencies------------------------------------------------------------------
add_dependencies(clingUtils clangCodeGen)
add_dependencies(clingInterpreter clangBasic clangAST ClangDriverOptions)
add_dependencies(clingMetaProcessor clangAST ClangDriverOptions)

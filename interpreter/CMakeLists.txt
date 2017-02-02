
#---Define the way we want to build and what of llvm/clang/cling------------------------------------
set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_FORCE_USE_OLD_TOOLCHAIN ON CACHE BOOL "")
# We only use llvm/clang through TCling which is (with the help of core/meta) already taking a lock
# to serialize access to llvm.  We can later review how to make this finer grained by using llvm's own locking
# mechanism.
set(LLVM_ENABLE_THREADS OFF CACHE BOOL "")
if(cxx14)
  set(LLVM_ENABLE_CXX1Y ON CACHE BOOL "" FORCE)
endif()

set(CMAKE_REQUIRED_QUIET 1)  # Make the configuration of LLVM quiet

if(ROOT_ARCHITECTURE MATCHES linuxarm64)
  set(ROOT_CLING_TARGET "AArch64")
elseif(ROOT_ARCHITECTURE MATCHES linuxarm)
  set(ROOT_CLING_TARGET "ARM")
elseif(ROOT_ARCHITECTURE MATCHES linuxppc64gcc)
  set(ROOT_CLING_TARGET "PowerPC")
elseif(ROOT_ARCHITECTURE MATCHES linuxs390)
  set(ROOT_CLING_TARGET "SystemZ")
elseif(ROOT_ARCHITECTURE MATCHES linux)
  set(ROOT_CLING_TARGET "X86")
elseif(ROOT_ARCHITECTURE MATCHES macosx)
  set(ROOT_CLING_TARGET "X86")
else()
  set(ROOT_CLING_TARGET "all")
endif()

set(LLVM_TARGETS_TO_BUILD ${ROOT_CLING_TARGET} CACHE STRING "Semicolon-separated list of targets to build, or \"all\".")

#---Protect against warnings clang/gcc would fire----------------------------------------------------
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-comment)
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-misleading-indentation)
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-missing-field-initializers)
if(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
  ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-conditional-uninitialized)
  ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-unused-private-field)
  ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-unused-local-typedef)
  ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-parentheses-equality)
  ROOT_ADD_C_FLAG(INTERP_ADDITIONAL_C_FLAGS -Wno-parentheses-equality)
endif()
if(APPLE)
  ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-deprecated-declarations)
endif()
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-overloaded-virtual)
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-maybe-uninitialized)
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-unused-parameter)
ROOT_ADD_C_FLAG(INTERP_ADDITIONAL_C_FLAGS -Wno-unused-parameter)
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-unused-variable)
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-unused-but-set-variable)
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-unused-function)
ROOT_ADD_CXX_FLAG(INTERP_ADDITIONAL_CXX_FLAGS -Wno-range-loop-analysis)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${INTERP_ADDITIONAL_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${INTERP_ADDITIONAL_C_FLAGS}")

#---Do not transform warnings in errors-------------------------------------------------------------
string(REPLACE "-Werror " "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
string(REPLACE "-Wshadow " "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})

#---Build LLVM/Clang with symbol visibility=hidden--------------------------------------------------
if(NOT CLING_INCLUDE_TESTS)
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -fvisibility=hidden)
endif()

#--- Build LLVM/Clang with modules -----------------------------------------------------------------
if(cxxmodules)
  # LLVM knows how to configure its modules builds. We cannot just add the flags
  # because the cxxmodules builds in llvm have different build dependency order.
  string(REPLACE "${ROOT_CXXMODULES_CXXFLAGS}" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  string(REPLACE "${ROOT_CXXMODULES_CFLAGS}" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
  if(libcxx)
    # FIXME: We cannot build LLVM/clang with modules on with libstdc++, yet.
    set (LLVM_ENABLE_MODULES ON CACHE BOOL "Override the default LLVM_ENABLE_MODULES option value." )
  endif(libcxx)
endif(cxxmodules)

#---Build type--------------------------------------------------------------------------------------
if(NOT DEFINED LLVM_BUILD_TYPE)
  set(LLVM_BUILD_TYPE Release CACHE STRING "Build type for LLVM (used to set CMAKE_BUILD_TYPE)")
endif()
if( CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT LLVM_BUILD_TYPE STREQUAL "Debug")
  message(WARNING "Selected a 'Debug' build (CMAKE_BUILD_TYPE), be aware that the embedded LLVM will still be built as 'Release'."
                  " Use the variable LLVM_BUILD_TYPE to control it.")
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
set(BUILD_SHARED_LIBS "NO")
#--- Do not build cling as part of llvm.
set(LLVM_NOCLING "YES")

set(CLANG_ENABLE_STATIC_ANALYZER OFF CACHE BOOL "")
set(CLANG_ENABLE_ARCMT OFF CACHE BOOL "")
set(CLANG_ENABLE_COMPILER OFF CACHE BOOL "")
set(CLANG_ENABLE_FORMAT OFF CACHE BOOL "")

#---Remove the inherited include_directories()
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "")

#---Add the sub-directory excluding all the targets from all-----------------------------------------
if(CMAKE_GENERATOR MATCHES "Xcode")
  add_subdirectory(llvm/src)
else()
  add_subdirectory(llvm/src EXCLUDE_FROM_ALL)
endif()

#---Avoid to dependencies to system (e.g. atomic) libraries without modifying the LLVM code----------
if(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  get_target_property(__libs LLVMSupport INTERFACE_LINK_LIBRARIES)
  list(REMOVE_ITEM __libs atomic)
  set_target_properties(LLVMSupport PROPERTIES INTERFACE_LINK_LIBRARIES "${__libs}")
endif()

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
set(llvmconfigfile "${CMAKE_CURRENT_SOURCE_DIR}/llvm/src/CMakeLists.txt")
file(READ ${llvmconfigfile} _filestr)
foreach(versionpart LLVM_VERSION_MAJOR LLVM_VERSION_MINOR LLVM_VERSION_PATCH)
  if(${_filestr} MATCHES "set[(](${versionpart}[^)]+)")
    string(REGEX REPLACE " *${versionpart} ([^)]+).*" "\\1" ${versionpart} "${CMAKE_MATCH_1}")
  else()
    MESSAGE(FATAL_ERROR "Cannot extract LLVM version number (${versionpart}) from ${llvmconfigfile}")
  endif()
endforeach()
set(LLVM_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}" PARENT_SCOPE)


#---Define the way we want to build and what of llvm/clang/cling------------------------------------
set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "")
set(CLANG_INCLUDE_TESTS OFF CACHE BOOL "")
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "")
set(CLANG_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_BUILD_TOOLS OFF CACHE BOOL "")
set(LLVM_TOOL_LLVM_AR_BUILD OFF CACHE BOOL "")
set(CLANG_TOOL_CLANG_OFFLOAD_BUNDLER_BUILD OFF CACHE BOOL "")
set(LLVM_FORCE_USE_OLD_TOOLCHAIN ON CACHE BOOL "")
set(LLVM_EXTERNAL_CLING_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cling" CACHE STRING "")
set(LLVM_EXTERNAL_PROJECTS "cling" CACHE STRING "")
# We only use llvm/clang through TCling which is (with the help of core/meta) already taking a lock
# to serialize access to llvm.  We can later review how to make this finer grained by using llvm's own locking
# mechanism.
set(LLVM_ENABLE_THREADS OFF CACHE BOOL "")
if(cxx17)
  set(LLVM_ENABLE_CXX1Z ON CACHE BOOL "" FORCE)
elseif(cxx14)
  set(LLVM_ENABLE_CXX1Y ON CACHE BOOL "" FORCE)
endif()

# The llvm::ReverseIterate<bool>::value symbol from llvm's SmallPtrSet.h
# somehow lands in our cling libraries on OS X and doesn't get hidden
# by visibility-inlines-hidden, so we suddenly have a global weak symbol
# from LLVM in cling which our visiblity=hidden compiled LLVM libraries
# reference. This is triggering some build system warnings like this:
#   ld: warning: direct access in function '(anonymous namespace)::NewGVN::runGVN()'
#   from file 'interpreter/llvm/src/lib/libLLVMScalarOpts.a(NewGVN.cpp.o)' to global weak symbol
#   'llvm::ReverseIterate<bool>::value' from file 'interpreter/llvm/src/lib/libclingUtils.a(AST.cpp.o)'
#   means the weak symbol cannot be overridden at runtime. This was likely caused by different
#   translation units being compiled with different visibility settings.
# There is no apparent reason why this is happening and it looks like a compiler bug,
# so let's just disable the part of the code that provides this symbol.
# As it's in the validation part of LLVM and not in something that providing functionality,
# this shouldn't cause any problems.
# TODO: We maybe can remove this code once we upgrade to LLVM>=6.0 as this symbol
# was introduced quite recently into LLVM 5.0 and probably is also causing problems
# for some other projects.
set(LLVM_ENABLE_ABI_BREAKING_CHECKS OFF CACHE BOOL "" FORCE)
set(LLVM_ABI_BREAKING_CHECKS FORCE_OFF CACHE BOOL "" FORCE)

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

if(clingtest)
  message("-- cling test suite enabled: llvm / clang symbols in libCling will be visible!")
  set(CLING_INCLUDE_TESTS ON CACHE BOOL "")
else()
  #---Build LLVM/Clang with symbol visibility=hidden--------------------------------------------------
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -fvisibility=hidden)
  set(CLANG_ENABLE_COMPILER OFF CACHE BOOL "")
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

if(gcctoolchain)
  ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS --gcc-toolchain=${gcctoolchain})
endif()

#---Build type--------------------------------------------------------------------------------------
if(NOT DEFINED LLVM_BUILD_TYPE)
  set(LLVM_BUILD_TYPE Release CACHE STRING "Build type for LLVM (used to set CMAKE_BUILD_TYPE)")
endif()
if( CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT LLVM_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Selected a 'Debug' build (CMAKE_BUILD_TYPE), be aware that the embedded LLVM will still be built as 'Release'."
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
set(CLANG_ENABLE_FORMAT OFF CACHE BOOL "")

#---Remove the inherited include_directories()
set_directory_properties(PROPERTIES INCLUDE_DIRECTORIES "")

#---Disable VCS writing to prevent that we always have to rebuild when someone touched the git HEAD
set(LLVM_APPEND_VC_REV OFF CACHE BOOL "Disabled LLVM revision dependency" FORCE)

#---Add the sub-directory excluding all the targets from all-----------------------------------------
if(CMAKE_GENERATOR MATCHES "Xcode")
  add_subdirectory(llvm/src)
else()
  add_subdirectory(llvm/src EXCLUDE_FROM_ALL)
endif()

#---Make LLVM compilation flags public---------------------------------------------------------------
# Extract the compilation flags from LLVM and make them public to the
# rest of ROOT so that we can compile against LLVM with matching flags.

# LLVM doesn't really give us a API to get this with an in-source build
# so we just use the normal way of doing this and read the llvm directory
# compilation properties.
get_directory_property(LLVM_DEFS DIRECTORY llvm/src COMPILE_DEFINITIONS)
# Turns DEFINE1;DEFINE2 to -DDEFINE1 -DDEFINE2
string (REPLACE ";" " -D" LLVM_DEFS ";${LLVM_DEFS}")
# CACHE variable that contains the additional flags that LLVM needs.
set(ROOT_LLVM_FLAGS "${LLVM_DEFS}" CACHE STRING "" FORCE)

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
  if("${_filestr}" MATCHES "set[(](${versionpart}[^)]+)")
    string(REGEX REPLACE " *${versionpart} ([^)]+).*" "\\1" ${versionpart} "${CMAKE_MATCH_1}")
  else()
    MESSAGE(FATAL_ERROR "Cannot extract LLVM version number (${versionpart}) from ${llvmconfigfile}")
  endif()
endforeach()
set(LLVM_VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}" PARENT_SCOPE)

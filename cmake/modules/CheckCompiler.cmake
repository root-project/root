#---------------------------------------------------------------------------------------------------
#  CheckCompiler.cmake
#---------------------------------------------------------------------------------------------------

include(CheckLanguage)
#---Enable FORTRAN (unfortunatelly is not not possible in all cases)-------------------------------
if(fortran)
  #--Work-around for CMake issue 0009220
  if(DEFINED CMAKE_Fortran_COMPILER AND CMAKE_Fortran_COMPILER MATCHES "^$")
    set(CMAKE_Fortran_COMPILER CMAKE_Fortran_COMPILER-NOTFOUND)
  endif()
  if(CMAKE_Fortran_COMPILER)
    # CMAKE_Fortran_COMPILER has already been defined somewhere else, so
    # just check whether it contains a valid compiler
    enable_language(Fortran)
  else()
    # CMAKE_Fortran_COMPILER has not been defined, so first check whether
    # there is a Fortran compiler at all
    check_language(Fortran)
    if(CMAKE_Fortran_COMPILER)
      # Fortran compiler found, however as 'check_language' was executed
      # in a separate process, the result might not be compatible with
      # the C++ compiler, so reset the variable, ...
      unset(CMAKE_Fortran_COMPILER CACHE)
      # ..., and enable Fortran again, this time prefering compilers
      # compatible to the C++ compiler
      enable_language(Fortran)
    endif()
  endif()
else()
  set(CMAKE_Fortran_COMPILER CMAKE_Fortran_COMPILER-NOTFOUND)
endif()

#----Get the compiler file name (to ensure re-location)---------------------------------------------
get_filename_component(_compiler_name ${CMAKE_CXX_COMPILER} NAME)
get_filename_component(_compiler_path ${CMAKE_CXX_COMPILER} PATH)
if("$ENV{PATH}" MATCHES ${_compiler_path})
  set(CXX ${_compiler_name})
else()
  set(CXX ${CMAKE_CXX_COMPILER})
endif()

#----Test if clang setup works----------------------------------------------------------------------
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  exec_program(${CMAKE_CXX_COMPILER} ARGS "--version 2>&1 | grep version" OUTPUT_VARIABLE _clang_version_info)
  string(REGEX REPLACE "^.*[ ]version[ ]([0-9]+)\\.[0-9]+.*" "\\1" CLANG_MAJOR "${_clang_version_info}")
  string(REGEX REPLACE "^.*[ ]version[ ][0-9]+\\.([0-9]+).*" "\\1" CLANG_MINOR "${_clang_version_info}")
  message(STATUS "Found Clang. Major version ${CLANG_MAJOR}, minor version ${CLANG_MINOR}")
  set(COMPILER_VERSION clang${CLANG_MAJOR}${CLANG_MINOR})
  if(CMAKE_GENERATOR STREQUAL "Ninja")
    # LLVM/Clang are automatically checking if we are in interactive terminal mode.
    # We use color output only for Ninja, because Ninja by default is buffering the output,
    # so Clang disables colors as it is sure whether the output goes to a file or to a terminal.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fcolor-diagnostics")
  endif()
  if(ccache AND CCACHE_VERSION VERSION_LESS "3.2.0")
    # https://bugzilla.samba.org/show_bug.cgi?id=8118
    # Call to 'ccache clang' is triggering next warning (valid for ccache 3.1.x, fixed in 3.2):
    # "clang: warning: argument unused during compilation: '-c"
    # Adding -Qunused-arguments provides a workaround for the bug.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qunused-arguments")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Qunused-arguments")
  endif()
else()
  set(CLANG_MAJOR 0)
  set(CLANG_MINOR 0)
endif()

#---Obtain the major and minor version of the GNU compiler-------------------------------------------
if (CMAKE_COMPILER_IS_GNUCXX)
  exec_program(${CMAKE_C_COMPILER} ARGS "-dumpversion" OUTPUT_VARIABLE _gcc_version_info)
  string(REGEX REPLACE "^([0-9]+).*$"                   "\\1" GCC_MAJOR ${_gcc_version_info})
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$"          "\\1" GCC_MINOR ${_gcc_version_info})
  string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" GCC_PATCH ${_gcc_version_info})

  if(GCC_PATCH MATCHES "\\.+")
    set(GCC_PATCH "")
  endif()
  if(GCC_MINOR MATCHES "\\.+")
    set(GCC_MINOR "")
  endif()
  if(GCC_MAJOR MATCHES "\\.+")
    set(GCC_MAJOR "")
  endif()
  message(STATUS "Found GCC. Major version ${GCC_MAJOR}, minor version ${GCC_MINOR}")
  set(COMPILER_VERSION gcc${GCC_MAJOR}${GCC_MINOR}${GCC_PATCH})
else()
  set(GCC_MAJOR 0)
  set(GCC_MINOR 0)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: Release, MinSizeRel, Debug, RelWithDebInfo." FORCE)
endif()

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

#---Check for cxx11 option------------------------------------------------------------
if(cxx11 AND cxx14)
  message(STATUS "c++11 mode requested but superseded by request for c++14 mode")
  set(cxx11 OFF CACHE BOOL "" FORCE)
endif()
if((cxx11 OR cxx14) AND cxx17)
  message(STATUS "c++11 or c++14 mode requested but superseded by request for c++17 mode")
  set(cxx11 OFF CACHE BOOL "" FORCE)
  set(cxx14 OFF CACHE BOOL "" FORCE)
endif()
if(cxx11)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CXX11)
  if(NOT HAS_CXX11)
    message(STATUS "Current compiler does not suppport -std=c++11 option. Switching OFF cxx11 option")
    set(cxx11 OFF CACHE BOOL "" FORCE)
  endif()
endif()
if(cxx14)
  CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CXX14)
  if(NOT HAS_CXX14)
    message(STATUS "Current compiler does not suppport -std=c++14 option. Switching OFF cxx14 option")
    set(cxx14 OFF CACHE BOOL "" FORCE)
  endif()
endif()
if(cxx17)
  CHECK_CXX_COMPILER_FLAG("-std=c++1z" HAS_CXX17)
  if(NOT HAS_CXX17)
    message(STATUS "Current compiler does not suppport -std=c++17 option. Switching OFF cxx17 option")
    set(cxx17 OFF CACHE BOOL "" FORCE)
  endif()
endif()
if(root7)
  if(cxx11)
    message(STATUS "ROOT7 interfaces require >= cxx14 which is disabled. Switching OFF root7 option")
    set(root7 OFF CACHE BOOL "" FORCE)
  endif()
    set(http ON CACHE BOOL "" FORCE)
endif()

#---Check for libcxx option------------------------------------------------------------
if(libcxx)
  CHECK_CXX_COMPILER_FLAG("-stdlib=libc++" HAS_LIBCXX11)
  if(NOT HAS_LIBCXX11)
    message(STATUS "Current compiler does not suppport -stdlib=libc++ option. Switching OFF libcxx option")
    set(libcxx OFF CACHE BOOL "" FORCE)
  endif()
endif()

#---Need to locate thead libraries and options to set properly some compilation flags----------------
find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  set(CMAKE_THREAD_FLAG -pthread)
else()
  set(CMAKE_THREAD_FLAG)
endif()


#---Setup compiler-specific flags (warning etc)----------------------------------------------
if(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
  # AppleClang and Clang proper.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wc++11-narrowing -Wsign-compare -Wsometimes-uninitialized -Wconditional-uninitialized -Wheader-guard -Warray-bounds -Wcomment -Wtautological-compare -Wstrncat-size -Wloop-analysis -Wbool-conversion")
elseif(CMAKE_COMPILER_IS_GNUCXX)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-implicit-fallthrough -Wno-noexcept-type")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-implicit-fallthrough")
  endif()
endif()


#---Setup details depending on the major platform type----------------------------------------------
if(CMAKE_SYSTEM_NAME MATCHES Linux)
  include(SetUpLinux)
elseif(APPLE)
  include(SetUpMacOS)
elseif(WIN32)
  include(SetupWindows)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_THREAD_FLAG}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_THREAD_FLAG}")

if(cxx11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if(cxx14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()

if(cxx17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
endif()

if(libcxx)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

if(gcctoolchain)
  CHECK_CXX_COMPILER_FLAG("--gcc-toolchain=${gcctoolchain}" HAS_GCCTOOLCHAIN)
  if(HAS_GCCTOOLCHAIN)
     set(CMAKE_CXX_FLAGS "--gcc-toolchain=${gcctoolchain} ${CMAKE_CXX_FLAGS}")
  endif()
endif()

if(gnuinstall)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DR__HAVE_CONFIG")
endif()

#---Check if we use the new libstdc++ CXX11 ABI-----------------------------------------------------
# Necessary to compile check_cxx_source_compiles this early
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
"
#include <string>
#if _GLIBCXX_USE_CXX11_ABI == 0
  #error NOCXX11
#endif
int main() {}
" GLIBCXX_USE_CXX11_ABI)

#---Print the final compiler flags--------------------------------------------------------------------
message(STATUS "ROOT Platform: ${ROOT_PLATFORM}")
message(STATUS "ROOT Architecture: ${ROOT_ARCHITECTURE}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}")

#---------------------------------------------------------------------------------------------------
#  CheckCompiler.cmake
#---------------------------------------------------------------------------------------------------

#---Enable FORTRAN (unfortunatelly is not not possible in all cases)-------------------------------
if(fortran)
  #--Work-around for CMake issue 0009220
  if(DEFINED CMAKE_Fortran_COMPILER AND CMAKE_Fortran_COMPILER MATCHES "^$")
    set(CMAKE_Fortran_COMPILER CMAKE_Fortran_COMPILER-NOTFOUND)
  endif()
  enable_language(Fortran OPTIONAL)
  if(NOT CMAKE_Fortran_COMPILER)
    set(CMAKE_Fortran_COMPILER_LOADED)   # FindBLAS/LAPACK tests CMAKE_Fortran_COMPILER_LOADED
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
  if(ccache)
    # https://bugzilla.samba.org/show_bug.cgi?id=8118 and color.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Qunused-arguments -fcolor-diagnostics")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Qunused-arguments -fcolor-diagnostics")
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

#---Set a default build type for single-configuration CMake generators if no build type is set------
if(WIN32)
  set(CMAKE_CONFIGURATION_TYPES Release MinSizeRel Debug RelWithDebInfo)
else()
  set(CMAKE_CONFIGURATION_TYPES Release MinSizeRel Debug RelWithDebInfo Optimized)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: Release, MinSizeRel, Debug, RelWithDebInfo, Optimized." FORCE)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
string(TOUPPER "${CMAKE_CONFIGURATION_TYPES}" uppercase_CMAKE_CONFIGURATION_TYPES)
if(NOT "${uppercase_CMAKE_CONFIGURATION_TYPES}" MATCHES "${uppercase_CMAKE_BUILD_TYPE}")
  message(FATAL_ERROR "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} needs to be one of known build types: ${CMAKE_CONFIGURATION_TYPES}")
endif()

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

#---Check for cxx11 option------------------------------------------------------------
if(cxx11 AND cxx14)
  message(STATUS "c++11 mode requested but superseded by request for c++14 mode")
  set(cxx11 OFF CACHE BOOL "" FORCE)
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
  set(root7 On)
endif()
if(root7)
  if(NOT cxx14)
    message(STATUS "ROOT7 interfaces require cxx14 which is disabled. Switching OFF root7 option")
    set(root7 OFF CACHE BOOL "" FORCE)
  endif()
endif()

#---Check for libcxx option------------------------------------------------------------
if(libcxx)
  CHECK_CXX_COMPILER_FLAG("-stdlib=libc++" HAS_LIBCXX11)
  if(NOT HAS_LIBCXX11)
    message(STATUS "Current compiler does not suppport -stdlib=libc++ option. Switching OFF libcxx option")
    set(libcxx OFF CACHE BOOL "" FORCE)
  endif()
endif()


#---Check for cxxmodules option------------------------------------------------------------
if(cxxmodules)
  # Copy-pasted from HandleLLVMOptions.cmake, please keep up to date.
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fmodules -fcxx-modules")
  # Check that we can build code with modules enabled, and that repeatedly
  # including <cassert> still manages to respect NDEBUG properly.
  CHECK_CXX_SOURCE_COMPILES("#undef NDEBUG
                             #include <cassert>
                             #define NDEBUG
                             #include <cassert>
                             int main() { assert(this code is not compiled); }"
                             CXX_SUPPORTS_MODULES)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  if(CXX_SUPPORTS_MODULES)
    file(COPY ${CMAKE_SOURCE_DIR}/build/unix/module.modulemap DESTINATION ${ROOT_INCLUDE_DIR})
    set(ROOT_CXXMODULES_COMMONFLAGS "-fmodules -fmodules-cache-path=${CMAKE_BINARY_DIR}/include/pcms/ -fno-autolink -fdiagnostics-show-note-include-stack -Rmodule-build")
    if (APPLE)
      # FIXME: TGLIncludes and alike depend on glew.h doing special preprocessor
      # trickery to override the contents of system's OpenGL.
      # On OSX #include TGLIncludes.h will trigger the creation of the system
      # OpenGL.pcm. Once it is built, glew cannot use preprocessor trickery to 'fix'
      # the translation units which it needs to 'rewrite'. The translation units
      # which need glew support are in graf3d. However, depending on the modulemap
      # organization we could request it implicitly (eg. one big module for ROOT).
      # In these cases we need to 'prepend' this include path to the compiler in order
      # for glew.h to it its trick.
      #set(ROOT_CXXMODULES_COMMONFLAGS "${ROOT_CXXMODULES_COMMONFLAGS} -isystem ${CMAKE_SOURCE_DIR}/graf3d/glew/isystem"

      # Workaround the issue described above by not picking up the system module
      # maps. In this way we can use the -fmodules-local-submodule-visibility
      # flag.
      set(ROOT_CXXMODULES_COMMONFLAGS "${ROOT_CXXMODULES_COMMONFLAGS} -fno-implicit-module-maps -fmodule-map-file=${CMAKE_BINARY_DIR}/include/module.modulemap")
    endif(APPLE)
    # This var is useful when we want to compile things without cxxmodules.
    set(ROOT_CXXMODULES_CXXFLAGS "${ROOT_CXXMODULES_COMMONFLAGS} -fcxx-modules -Xclang -fmodules-local-submodule-visibility" CACHE STRING "Useful to filter out the modules-related cxxflags.")

    set(ROOT_CXXMODULES_CFLAGS "${ROOT_CXXMODULES_COMMONFLAGS}" CACHE STRING "Useful to filter out the modules-related cflags.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ROOT_CXXMODULES_CFLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ROOT_CXXMODULES_CXXFLAGS}")
  else()
    message(FATAL_ERROR "cxxmodules is not supported by this compiler")
  endif()
endif(cxxmodules)

#---Need to locate thead libraries and options to set properly some compilation flags----------------
find_package(Threads)
if(CMAKE_USE_PTHREADS_INIT)
  set(CMAKE_THREAD_FLAG -pthread)
else()
  set(CMAKE_THREAD_FLAG)
endif()

#---Setup details depending opn the major platform type----------------------------------------------
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
  if(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wc++11-narrowing -Wsign-compare -Wsometimes-uninitialized -Wconditional-uninitialized -Wheader-guard -Warray-bounds -Wcomment -Wtautological-compare -Wstrncat-size -Wloop-analysis -Wbool-conversion")
  endif()
endif()

if(cxx14)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
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

#---Print the final compiler flags--------------------------------------------------------------------
message(STATUS "ROOT Platform: ${ROOT_PLATFORM}")
message(STATUS "ROOT Architecture: ${ROOT_ARCHITECTURE}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}")

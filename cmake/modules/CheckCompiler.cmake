#---------------------------------------------------------------------------------------------------
#  CheckCompiler.cmake
#---------------------------------------------------------------------------------------------------

#---Enable FORTRAN (unfortunatelly is not nowt possible in all cases)-------------------------------
if(NOT WIN32 AND NOT CMAKE_GENERATOR STREQUAL Xcode)
  enable_language(Fortran OPTIONAL)
endif()


#----Test if clang setup works----------------------------------------------------------------------
if(CMAKE_C_COMPILER MATCHES clang)
  exec_program(${CMAKE_C_COMPILER} ARGS "-v" OUTPUT_VARIABLE _clang_version_info)
  string(REGEX REPLACE "^.*[ ]([0-9]+)\\.[0-9].*$" "\\1" CLANG_MAJOR "${_clang_version_info}")
  string(REGEX REPLACE "^.*[ ][0-9]+\\.([0-9]).*$" "\\1" CLANG_MINOR "${_clang_version_info}")
else()
  set(CLANG_MAJOR 0)
  set(CLANG_MINOR 0)
endif()

#---Obtain the major and minor version of the GNU compiler-------------------------------------------
if (CMAKE_COMPILER_IS_GNUCXX)
  exec_program(${CMAKE_C_COMPILER} ARGS "-dumpversion" OUTPUT_VARIABLE _gcc_version_info)
  string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\.[0-9]" "\\1" GCC_MAJOR "${_gcc_version_info}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+)\\.[0-9]" "\\1" GCC_MINOR "${_gcc_version_info}")
  string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9])" "\\1" GCC_PATCH "${_gcc_version_info}")
  message(STATUS "Found GCC. Major version ${GCC_MAJOR}, minor version ${GCC_MINOR}")
  set(COMPILER_VERSION gcc${GCC_MAJOR}${GCC_MINOR}${GCC_PATCH})
else()
  set(GCC_MAJOR 0)
  set(GCC_MINOR 0)
endif()

#---Set a default build type for single-configuration CMake generators if no build type is set------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

#---Need to locate thead libraries and options to set properly some compilation flags---------------- 
find_package(Threads)

#---Setup details depending opn the major platform type----------------------------------------------
if(CMAKE_SYSTEM_NAME MATCHES Linux)
  include(SetUpLinux)
elseif(APPLE)
  include(SetUpMacOS)
elseif(WIN32)
  include(SetupWindows)
endif()

#---Print the final compiler flags--------------------------------------------------------------------
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler Flags: ${CMAKE_CXX_FLAGS} ${ALL_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

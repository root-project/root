# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(ROOT_PLATFORM macosx)

if (CMAKE_SYSTEM_NAME MATCHES Darwin)
  MESSAGE(STATUS "Found a macOS system")

  if(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
    set(libcxx ON CACHE BOOL "Build using libc++" FORCE)
  endif()

  #TODO: check haveconfig and rpath -> set rpath true
  #TODO: check Thread, define link command
  #TODO: more stuff check configure script
  if(CMAKE_SYSTEM_PROCESSOR MATCHES arm64)
     MESSAGE(STATUS "Found an AArch64 system")
     set(ROOT_ARCHITECTURE macosxarm64)
  else()
     MESSAGE(STATUS "Found an x86_64 system")
     set(ROOT_ARCHITECTURE macosx64)
     SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -m64")
  endif()

  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -m64")
  SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -m64")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")

  set(MACOSX_SSL_DEPRECATED ON)
  set(MACOSX_GLU_DEPRECATED ON)

  if (CMAKE_COMPILER_IS_GNUCXX)
     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -W -Wshadow -Wall -Woverloaded-virtual -fsigned-char -fsized-deallocation -fno-common")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -W -Wall -fsigned-char -fno-common")
     SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")

     SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -Wl,-dead_strip_dylibs")
     SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -Wl,-dead_strip_dylibs")

     set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -m64")
     set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -m64")

  elseif(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -W -Wall -Woverloaded-virtual -fsigned-char -fsized-deallocation -fno-common -Qunused-arguments")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -W -Wall -fsigned-char -fno-common -Qunused-arguments")
     if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
     endif()

     SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")

     SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -Wl,-dead_strip_dylibs")
     SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -Wl,-dead_strip_dylibs")

     set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -m64")
     set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -m64")

     if(asan)
       # See also core/sanitizer/README.md for what's happening.
       execute_process(COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan_osx_dynamic.dylib OUTPUT_VARIABLE ASAN_RUNTIME_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
       set(ASAN_EXTRA_CXX_FLAGS -fsanitize=address -fno-omit-frame-pointer -fsanitize-address-use-after-scope)
       set(ASAN_EXTRA_SHARED_LINKER_FLAGS "-fsanitize=address")
       set(ASAN_EXTRA_EXE_LINKER_FLAGS "-fsanitize=address -Wl,-U,__asan_default_options -Wl,-U,__lsan_default_options -Wl,-U,__lsan_default_suppressions")
     endif()

  else()
    MESSAGE(FATAL_ERROR "There is no setup for this compiler with ID=${CMAKE_CXX_COMPILER_ID} up to now. Don't know what to do. Stop cmake at this point.")
  endif()
else (CMAKE_SYSTEM_NAME MATCHES Darwin)
  MESSAGE(FATAL_ERROR "There is no setup for this this Apple system up to now. Don't know waht to do. Stop cmake at this point.")
endif (CMAKE_SYSTEM_NAME MATCHES Darwin)

#---Avoid puting the libraires and executables in different configuration locations
if(CMAKE_GENERATOR MATCHES Xcode)
  foreach( _conf ${CMAKE_CONFIGURATION_TYPES} )
    string( TOUPPER ${_conf} _conf )
    set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_conf} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
    set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_conf} ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} )
    set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_conf} ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY} )
  endforeach()
endif()

#---Avoid using a x86_64 Ninja executable with on a arm64 MacOS
#---This issue leads to the external being build for x86_64 instead of arm64
execute_process(COMMAND lipo -archs ${CMAKE_MAKE_PROGRAM} OUTPUT_VARIABLE _NINJA_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
if(CMAKE_GENERATOR MATCHES Ninja)

  set( _NINJA_ARCH_LIST ${_NINJA_ARCH} )
  separate_arguments(_NINJA_ARCH_LIST) # This replace space with semi-colons
  if (NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" IN_LIST _NINJA_ARCH_LIST)
    message(FATAL_ERROR
            " ${CMAKE_MAKE_PROGRAM} does not support ${CMAKE_HOST_SYSTEM_PROCESSOR}.\n"
            " It only supports ${_NINJA_ARCH_LIST}.\n"
            " Downloading the latest version of Ninja might solve the problem.\n")
  endif()
endif()

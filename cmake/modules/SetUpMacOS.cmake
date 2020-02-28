# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

set(ROOT_ARCHITECTURE macosx)
set(ROOT_PLATFORM macosx)

if (CMAKE_SYSTEM_NAME MATCHES Darwin)
  EXECUTE_PROCESS(COMMAND sw_vers "-productVersion"
                  COMMAND cut -d . -f 1-2
                  OUTPUT_VARIABLE MACOSX_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

  MESSAGE(STATUS "Found a Mac OS X System ${MACOSX_VERSION}")

  if(MACOSX_VERSION VERSION_GREATER 10.7 AND ${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
    set(libcxx ON CACHE BOOL "Build using libc++" FORCE)
  endif()

  if(MACOSX_VERSION VERSION_GREATER 10.4)
    #TODO: check haveconfig and rpath -> set rpath true
    #TODO: check Thread, define link command
    #TODO: more stuff check configure script
    if(CMAKE_SYSTEM_PROCESSOR MATCHES 64)
       MESSAGE(STATUS "Found a 64bit system")
       set(ROOT_ARCHITECTURE macosx64)
       SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
       SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -m64")
       SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -m64")
       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
       SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -m64")
    else()
       MESSAGE(STATUS "Found a 32bit system")
       SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
       SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
       SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -m32")
    endif()
  endif()

  if(MACOSX_VERSION VERSION_GREATER 10.6)
    set(MACOSX_SSL_DEPRECATED ON)
  endif()
  if(MACOSX_VERSION VERSION_GREATER 10.7)
    set(MACOSX_ODBC_DEPRECATED ON)
  endif()
  if(MACOSX_VERSION VERSION_GREATER 10.8)
    set(MACOSX_GLU_DEPRECATED ON)
  endif()

  if (CMAKE_COMPILER_IS_GNUCXX)
     message(STATUS "Found GNU compiler collection")

     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -W -Wshadow -Wall -Woverloaded-virtual -fsigned-char -fno-common")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -W -Wall -fsigned-char -fno-common")
     SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")

     SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -single_module -Wl,-dead_strip_dylibs")
     SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -single_module -Wl,-dead_strip_dylibs")

     set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -bind_at_load -m64")
     set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -bind_at_load -m64")

     # Select flags.
     set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
     set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG")
     set(CMAKE_CXX_FLAGS_DEBUG          "-g")
     set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -DNDEBUG")
     set(CMAKE_C_FLAGS_RELEASE          "-O2 -DNDEBUG")
     set(CMAKE_C_FLAGS_DEBUG            "-g")
  elseif(${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
     message(STATUS "Found LLVM compiler collection")

     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe -W -Wall -Woverloaded-virtual -fsigned-char -fno-common -Qunused-arguments")
     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pipe -W -Wall -fsigned-char -fno-common -Qunused-arguments")
     if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
     endif()

     SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -std=legacy")

     SET(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS} -single_module -Wl,-dead_strip_dylibs")
     SET(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -single_module -Wl,-dead_strip_dylibs")

     set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} -bind_at_load -m64")
     set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -bind_at_load -m64")
     if(asan)
       #This should be the right way to do it, but clang 10 seems to have a bug
       #execute_process(COMMAND ${CMAKE_CXX_COMPILER} --print-file-name=libclang_rt.asan_osx_dynamic.dylib OUTPUT_VARIABLE ASAN_RUNTIME_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
       execute_process(COMMAND mdfind -name libclang_rt.asan_osx_dynamic.dylib OUTPUT_VARIABLE ASAN_RUNTIME_LIBRARY OUTPUT_STRIP_TRAILING_WHITESPACE)
       string(REGEX REPLACE "/[^/]+$" "" ASAN_RUNTIME_LIBRARY_PATH ${ASAN_RUNTIME_LIBRARY})
       set(ENV{${ld_library_path}} $ENV{${ld_library_path}}:${ASAN_RUNTIME_LIBRARY_PATH})
       if(lsan)
         set(ASAN_DETECT_LEAKS "1")
         set(ASAN_FAST_UNWIND "0")
       else()
         set(ASAN_DETECT_LEAKS "0")
         set(ASAN_FAST_UNWIND "1")
       endif()
       set(ENV{ASAN_OPTIONS} "fast_unwind_on_malloc=${ASAN_FAST_UNWIND}:strict_string_checks=1:detect_stack_use_after_return=1:check_initialization_order=1:detect_leaks=${ASAN_DETECT_LEAKS}:detect_container_overflow=1")
       set(ENV{LSAN_OPTIONS} "max_leaks=5:suppressions=${CMAKE_SOURCE_DIR}/build/LSan.supp:print_suppressions=0")

       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-blacklist=${CMAKE_SOURCE_DIR}/build/ASan_blacklist.txt -fno-omit-frame-pointer")
       set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -shared-libasan")
       set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -shared-libasan")
     endif()

     # Select flags.
     set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
     set(CMAKE_CXX_FLAGS_RELEASE        "-O2 -DNDEBUG")
     set(CMAKE_CXX_FLAGS_DEBUG          "-g")
     set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -DNDEBUG")
     set(CMAKE_C_FLAGS_RELEASE          "-O2 -DNDEBUG")
     set(CMAKE_C_FLAGS_DEBUG            "-g")
  else()
    MESSAGE(FATAL_ERROR "There is no setup for this compiler with ID=${CMAKE_CXX_COMPILER_ID} up to now. Don't know what to do. Stop cmake at this point.")
  endif()

  #---Set Linker flags----------------------------------------------------------------------
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mmacosx-version-min=${MACOSX_VERSION}")
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

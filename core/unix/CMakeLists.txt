# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/unix package
############################################################################

if(NOT UNIX)
  return()
endif()

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS TUnixSystem.h)
target_sources(Core PRIVATE src/TUnixSystem.cxx)
target_include_directories(Core PRIVATE inc ../clib/res)

if (CMAKE_SYSTEM_NAME MATCHES FreeBSD)
  target_link_libraries(Core PRIVATE execinfo util)
endif()

CHECK_CXX_SOURCE_COMPILES("#include <stdlib.h>
int main() { char buf[32]; arc4random_buf(buf, 32); return 0;}" found_arc4)

if(found_arc4)
   message(STATUS "Found arc4random_buf in stdlib.h")
   target_compile_definitions(Core PRIVATE R__ARC4_STDLIB)
else()
  CHECK_CXX_SOURCE_COMPILES("#include <bsd/stdlib.h>
  int main() { char buf[32]; arc4random_buf(buf, 32); return 0;}" found_arc4_bsd)
  if(found_arc4_bsd)
     message(STATUS "Found arc4random_buf in bsd/stdlib.h")
     target_compile_definitions(Core PRIVATE R__ARC4_BSDLIB)
  else()
    CHECK_CXX_SOURCE_COMPILES("#include <sys/random.h>
    int main() { char buf[32]; int res = getrandom(buf, 32, GRND_NONBLOCK); return 0;}" found_getrandom)
    if(found_getrandom)
      message(STATUS "Found getrandom in sys/random.h")
      target_compile_definitions(Core PRIVATE R__GETRANDOM_CLIB)
    else()
      message(FATAL_ERROR "Fail to detect cryptographic random generator")
    endif()
  endif()
endif()


ROOT_INSTALL_HEADERS()

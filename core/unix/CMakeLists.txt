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
  set(OLD_CMAKE_REQUIRED_INCLUDES ${CMAKE_REQUIRED_INCLUDES})
  set(OLD_CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES})
  if(DEFINED LIBBSDROOT)
     set(CMAKE_REQUIRED_INCLUDES ${LIBBSDROOT}/include)
     set(CMAKE_REQUIRED_LIBRARIES ${LIBBSDROOT}/lib/libbsd.so)
  endif()
  CHECK_CXX_SOURCE_COMPILES("#include <bsd/stdlib.h>
  int main() { char buf[32]; arc4random_buf(buf, 32); return 0;}" found_arc4_bsd)
  set(CMAKE_REQUIRED_INCLUDES ${OLD_CMAKE_REQUIRED_INCLUDES})
  set(CMAKE_REQUIRED_LIBRARIES ${OLD_CMAKE_REQUIRED_LIBRARIES})
  if(found_arc4_bsd)
    message(STATUS "Found arc4random_buf in bsd/stdlib.h")
    target_compile_definitions(Core PRIVATE R__ARC4_BSDLIB)
    if(DEFINED LIBBSDROOT)
      target_include_directories(Core PRIVATE ${LIBBSDROOT}/include)
      target_link_libraries(Core PRIVATE ${LIBBSDROOT}/lib/libbsd.so)
    endif()
  else()
    CHECK_CXX_SOURCE_COMPILES("#include <sys/random.h>
    int main() { char buf[32]; int res = getrandom(buf, 32, GRND_NONBLOCK); return 0;}" found_getrandom)
    if(found_getrandom)
      message(STATUS "Found getrandom in sys/random.h")
      target_compile_definitions(Core PRIVATE R__GETRANDOM_CLIB)
    else()
      CHECK_CXX_SOURCE_RUNS("
      #include <fstream>

      int main() {
          std::ifstream urandom{\"/dev/urandom\"};
          if (!urandom) {
              // This will make the CMake command fail
              return 1;
          }
      
          constexpr int len{32};
          char buf[len];
          for (int n = 0; n < len; n++) buf[n] = 0;
          urandom.read(buf, len);
      
          int nmatch = 0;
          for (int n = 0; n < len; n++)
              if (buf[n] == 0) nmatch++;
      
          // Fail if no values have changed
          return nmatch != len ? 0 : 1;
      }" found_urandom)
      if(found_urandom)
        message(STATUS "Found random device in /dev/urandom")
        target_compile_definitions(Core PRIVATE R__USE_URANDOM)
      else()
        message(FATAL_ERROR "Fail to detect cryptographic random generator")
      endif()
    endif()
  endif()
endif()


ROOT_INSTALL_HEADERS()

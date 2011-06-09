# Find the AfterImage includes and libraries.
#  (See http://afterstep.sourceforge.net/afterimage/)
# This module defines
# AFTERIMAGE_INCLUDE_DIR, where to locate libAfterImage header files
# AFTERIMAGE_LIBRARIES, the libraries to link against to use libAfterImage
# AFTERIMAGE_FOUND. If false, you cannot build anything that requires libAfterImage

if(AFTERIMAGE_CONFIG_EXECUTABLE)
  set(AFTERIMAGE_FIND_QUIETLY 1)
endif()
set(AFTERIMAGE_FOUND 0)

find_program(AFTERIMAGE_CONFIG_EXECUTABLE afterimage-config)

if(AFTERIMAGE_CONFIG_EXECUTABLE)
  execute_process(COMMAND ${AFTERIMAGE_CONFIG_EXECUTABLE} --version OUTPUT_VARIABLE AFTERIMAGE_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
  #---TODO (check that the version is sufficient)
  execute_process(COMMAND ${AFTERIMAGE_CONFIG_EXECUTABLE} --cflags OUTPUT_VARIABLE AFTERIMAGE_CFLAGS)
  string( REGEX MATCHALL "-I[^;]+" AFTERIMAGE_INCLUDE_DIR "${AFTERIMAGE_CFLAGS}" )
  string( REPLACE "-I" "" AFTERIMAGE_INCLUDE_DIR "${AFTERIMAGE_INCLUDE_DIRS}")
  execute_process(COMMAND ${AFTERIMAGE_CONFIG_EXECUTABLE} --libs OUTPUT_VARIABLE AFTERIMAGE_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(AFTERIMAGE_FOUND 1)  
endif()

if(AFTERIMAGE_FOUND)
  if(NOT AFTERIMAGE_FIND_QUIETLY)
    message(STATUS "Found AfterImage version ${AFTERIMAGE_VERSION} using ${AFTERIMAGE_CONFIG_EXECUTABLE}")
  endif()
endif()

mark_as_advanced(AFTERIMAGE_CONFIG_EXECUTABLE)

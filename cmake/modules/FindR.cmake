#Cmake module to find R 
# - Try to find R
# Once done, this will define
#
#  R_FOUND - system has R
#  R_INCLUDE_DIRS - the R include directories
#  R_LIBRARIES - link these to use R
#Autor: Omar Andres Zapata Mesa 31/05/2013

message(STATUS "Looking for R")

find_program ( R_EXECUTABLE
               NAMES R R.exe
              )
#searching flags unsing R executable              
if ( R_EXECUTABLE )
  execute_process ( COMMAND echo "cat(Sys.getenv(\"R_HOME\"))"
                    COMMAND ${R_EXECUTABLE} --vanilla --slave
                    OUTPUT_VARIABLE R_HOME
                    ERROR_VARIABLE R_HOME_ERR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                  )

 execute_process ( COMMAND ${R_EXECUTABLE} CMD config --cppflags
                   OUTPUT_VARIABLE R_INCLUDE_DIR
                   ERROR_VARIABLE  R_INCLUDE_DIR_ERR
                   OUTPUT_STRIP_TRAILING_WHITESPACE
                  )
                  
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(R_LIBRARY "-L${R_HOME}/lib -lR")
else()
  execute_process ( COMMAND ${R_EXECUTABLE} CMD config --ldflags
                    OUTPUT_VARIABLE R_LIBRARY
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                  )
endif()

# MESSAGE(STATUS "R_HOME=${R_HOME}")
# MESSAGE(STATUS "R_INCLUDE_DIR=${R_INCLUDE_DIR}")
# MESSAGE(STATUS "R_LIBRARY=${R_LIBRARY}")

else()              
set(R_PKGCONF_INCLUDE_DIRS  
      "/usr/local/include" "/usr/include" 
      "/opt/R/include" "/usr/share/R/include"  
      "/usr/local/share/R/include")

# Include dir
find_path(R_INCLUDE_DIR
  NAMES R.h
  PATHS ${R_PKGCONF_INCLUDE_DIRS}
)

set(R_PKGCONF_LIBRARY_DIRS  "/usr/local/lib" "/usr/lib"
    "/opt/R/lib" "/usr/lib/R/lib/" "/usr/lib/R/lib/" )
# Finally the library itself
find_library(R_LIBRARY
  NAMES R
  PATHS ${R_PKGCONF_LIBRARY_DIRS}
)

endif ( R_EXECUTABLE )

# Setting up the results 
set(R_INCLUDE_DIRS ${R_INCLUDE_DIR})
set(R_LIBRARIES ${R_LIBRARY})
if (("${R_INCLUDE_DIR}" STREQUAL "") OR ("${R_LIBRARY}" STREQUAL ""))
  set(R_FOUND FALSE)
  message(STATUS "Looking for R -- not found ")
else()
  set(R_FOUND TRUE)
  message(STATUS "Looking for R -- found")
endif()

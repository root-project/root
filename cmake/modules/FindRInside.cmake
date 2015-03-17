#Cmake module to find RInisde
# - Try to find Rinside
# Once done, this will define
#
#  RINSIDE_FOUND - system has RINSIDE
#  RINSIDE_INCLUDE_DIRS - the RINSIDE include directories
#  RINSIDE_LIBRARIES - link these to use RINSIDE
#Autor: Omar Andres Zapata Mesa 31/05/2013

message(STATUS "Looking for RInside")
find_program ( R_EXECUTABLE
               NAMES R R.exe
              )
if(R_EXECUTABLE)
  execute_process ( COMMAND echo "RInside:::CxxFlags()"
                    COMMAND ${R_EXECUTABLE} --vanilla --slave
                    OUTPUT_VARIABLE RINSIDE_INCLUDE_DIR
                    ERROR_VARIABLE RINSIDE_INCLUDE_DIR_ERR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                  )
  
  execute_process ( COMMAND echo "RInside:::LdFlags(static=0)"
                    COMMAND ${R_EXECUTABLE} --vanilla --slave
                    OUTPUT_VARIABLE RINSIDE_LIBRARY
                    ERROR_VARIABLE RINSIDE_LIBRARY_ERR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                  )

else()
set(RINSIDE_PKGCONF_INCLUDE_DIRS  
        "/usr/local/include /usr/include"
        "/opt/R/site-library/RInside/include"
        "/usr/local/lib/R/site-library/RInside/include"
        "/usr/lib/R/site-library/RInside/include")

# Include dir
find_path(RINSIDE_INCLUDE_DIR
  NAMES RInside.h
  PATHS ${RINSIDE_PKGCONF_INCLUDE_DIRS}
)

set(RINSIDE_PKGCONF_LIBRARY_DIRS  
	"/usr/local/lib" "/usr/lib"
        "/opt/R/site-library/RInside/lib"  "/usr/local/lib/R/site-library/RInside/lib"
        "/usr/lib/R/site-library/RInside/lib" )

# Finally the library itself
find_library(RINSIDE_LIBRARY
  NAMES libRInside.a libRInside.lib
  PATHS ${RINSIDE_PKGCONF_LIBRARY_DIRS}
)
endif(R_EXECUTABLE)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this  lib depends on.
set(RINSIDE_INCLUDE_DIRS ${RINSIDE_INCLUDE_DIR})
set(RINSIDE_LIBRARIES ${RINSIDE_LIBRARY})
if (("${RINSIDE_INCLUDE_DIR}" STREQUAL "") OR ("${RINSIDE_LIBRARY}" STREQUAL "")) 
  set(RINSIDE_FOUND FALSE)
  message(STATUS "Looking for RInside -- not found")
  message(STATUS "Install it running 'R -e \"install.packages(\\\"RInside\\\",repos=\\\"http://cran.irsn.fr/\\\")\"'")
else()
  set(RINSIDE_FOUND TRUE)
  message(STATUS "Looking for RInside -- found")
endif()

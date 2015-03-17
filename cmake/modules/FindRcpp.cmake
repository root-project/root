#Cmake module to find Rcpp
# - Try to find Rcpp
# Once done, this will define
#
#  RCPP_FOUND - system has RCPP
#  RCPP_INCLUDE_DIRS - the RCPP include directories
#  RCPP_LIBRARIES - link these to use RCPP
#Autor: Omar Andres Zapata Mesa 31/05/2013

message(STATUS "Looking for Rcpp")

find_program ( R_EXECUTABLE
               NAMES R R.exe
              )
execute_process ( COMMAND echo "Rcpp:::CxxFlags()"
                    COMMAND ${R_EXECUTABLE} --vanilla --slave
                    OUTPUT_VARIABLE RCPP_INCLUDE_DIR
                    ERROR_VARIABLE RCPP_INCLUDE_DIR_ERR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                  )
  
execute_process ( COMMAND echo "cat(find.package('Rcpp'))"
                    COMMAND ${R_EXECUTABLE} --vanilla --slave
                    OUTPUT_VARIABLE RCPP_PATH
                    ERROR_VARIABLE RCPP_PATH_ERR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                  )
                  
                  

set(RCPP_PKGCONF_LIBRARY_DIRS  
	"/usr/local/lib" "/usr/lib"
        "/opt/R/site-library/Rcpp/lib" 
        "/usr/local/lib/R/site-library/Rcpp/lib"
        "/usr/lib/R/site-library/Rcpp/lib"
        "${RCPP_PATH}/libs"
        )

# Finally the library itself
find_library(RCPP_LIBRARY
  NAMES libRcpp.a libRcpp.so  libRcpp.lib Rcpp.so libRcpp.dll Rcpp.dll Rcpp.dylib libRcpp.dylib
  PATHS ${RCPP_PKGCONF_LIBRARY_DIRS}
)


MESSAGE(STATUS "RCPP_INCLUDE_DIR=${RCPP_INCLUDE_DIR}")
MESSAGE(STATUS "RCPP_LIBRARY=${RCPP_LIBRARY}")

# Setting up the results 
set(RCPP_INCLUDE_DIRS ${RCPP_INCLUDE_DIR})
set(RCPP_LIBRARIES ${RCPP_LIBRARY})
if (("${RCPP_INCLUDE_DIR}" STREQUAL "") OR ("${RCPP_LIBRARY}" STREQUAL "")) 
  set(RCPP_FOUND FALSE)
  message(STATUS "Looking for Rcpp -- not found ")
  message(STATUS "Install it running 'R -e \"install.packages(\\\"RInside\\\",repos=\\\"http://cran.irsn.fr/\\\")\"'")
else()
  set(RCPP_FOUND TRUE)
  message(STATUS "Looking for Rcpp -- found")
endif()

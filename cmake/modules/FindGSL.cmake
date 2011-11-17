# Try to find gnu scientific library GSL
# See
# http://www.gnu.org/software/gsl/  and
# http://gnuwin32.sourceforge.net/packages/gsl.htm
#
# Based on a script of Felix Woelk and Jan Woetzel
# (www.mip.informatik.uni-kiel.de)
#
# It defines the following variables:
#  GSL_FOUND - system has GSL lib
#  GSL_INCLUDE_DIRS - where to find headers
#  GSL_LIBRARIES - full path to the libraries
#  GSL_LIBRARY_DIRS, the directory where the PLplot library is found.
#  GSL_CFLAGS, additional c (c++) required
 
set( GSL_FOUND OFF )
set( GSL_CBLAS_FOUND OFF )

if(GSL_INCLUDE_DIR OR GSL_CONFIG_EXECUTABLE)
  set(GSL_FIND_QUIETLY 1)
endif()

# Windows, but not for Cygwin and MSys where gsl-config is available
if( WIN32 AND NOT CYGWIN AND NOT MSYS )
  # look for headers
  find_path( GSL_INCLUDE_DIR
    NAMES gsl/gsl_cdf.h gsl/gsl_randist.h
	PATHS $ENV{GSL_DIR}/include
    )
  if( GSL_INCLUDE_DIR )
    # look for gsl library
    find_library( GSL_LIBRARY
      NAMES gsl
	  PATHS $ENV{GSL_DIR}/lib
    )
    if( GSL_LIBRARY )
      set( GSL_INCLUDE_DIRS ${GSL_INCLUDE_DIR} )
      get_filename_component( GSL_LIBRARY_DIRS ${GSL_LIBRARY} PATH )
      set( GSL_FOUND ON )
    endif( GSL_LIBRARY )
 
    # look for gsl cblas library
    find_library( GSL_CBLAS_LIBRARY
        NAMES gslcblas
		PATHS $ENV{GSL_DIR}/lib
      )
    if( GSL_CBLAS_LIBRARY )
      set( GSL_CBLAS_FOUND ON )
    endif( GSL_CBLAS_LIBRARY )
 
    set( GSL_LIBRARIES ${GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} )
	set( GSL_CFLAGS "-DGSL_DLL")
  endif( GSL_INCLUDE_DIR )
 
  mark_as_advanced(
    GSL_INCLUDE_DIR
    GSL_LIBRARY
    GSL_CBLAS_LIBRARY
  )
else( WIN32 AND NOT CYGWIN AND NOT MSYS )
  if( UNIX OR MSYS )
    find_program( GSL_CONFIG_EXECUTABLE gsl-config
      /usr/bin/
      /usr/local/bin
      $ENV{GSL_DIR}/bin
      ${GSL_DIR}/bin
    )
 
    if( GSL_CONFIG_EXECUTABLE )
      set( GSL_FOUND ON )
 
      # run the gsl-config program to get cxxflags
      execute_process(
        COMMAND sh "${GSL_CONFIG_EXECUTABLE}" --cflags
        OUTPUT_VARIABLE GSL_CFLAGS
        RESULT_VARIABLE RET
        ERROR_QUIET
        )
      if( RET EQUAL 0 )
        string( STRIP "${GSL_CFLAGS}" GSL_CFLAGS )
        separate_arguments( GSL_CFLAGS )
 
        # parse definitions from cflags; drop -D* from CFLAGS
        string( REGEX MATCHALL "-D[^;]+"
          GSL_DEFINITIONS  "${GSL_CFLAGS}" )
        string( REGEX REPLACE "-D[^;]+;" ""
          GSL_CFLAGS "${GSL_CFLAGS}" )
 
        # parse include dirs from cflags; drop -I prefix
        string( REGEX MATCHALL "-I[^;]+"
          GSL_INCLUDE_DIRS "${GSL_CFLAGS}" )
        string( REPLACE "-I" ""
          GSL_INCLUDE_DIRS "${GSL_INCLUDE_DIRS}")
        string( REGEX REPLACE "-I[^;]+;" ""
          GSL_CFLAGS "${GSL_CFLAGS}")
      else( RET EQUAL 0 )
        set( GSL_FOUND FALSE )
      endif( RET EQUAL 0 )
 
      # run the gsl-config program to get the libs
      execute_process(
        COMMAND sh "${GSL_CONFIG_EXECUTABLE}" --libs
        OUTPUT_VARIABLE GSL_LIBRARIES
        RESULT_VARIABLE RET
        ERROR_QUIET
        )
      if( RET EQUAL 0 )
        string(STRIP "${GSL_LIBRARIES}" GSL_LIBRARIES )
        separate_arguments( GSL_LIBRARIES )
 
        # extract linkdirs (-L) for rpath (i.e., LINK_DIRECTORIES)
        string( REGEX MATCHALL "-L[^;]+"
          GSL_LIBRARY_DIRS "${GSL_LIBRARIES}" )
        string( REPLACE "-L" ""
          GSL_LIBRARY_DIRS "${GSL_LIBRARY_DIRS}" )
      else( RET EQUAL 0 )
        set( GSL_FOUND FALSE )
      endif( RET EQUAL 0 )
 
      MARK_AS_ADVANCED(
        GSL_CFLAGS
      )
      if(NOT GSL_FIND_QUIETLY)
        execute_process(
          COMMAND sh "${GSL_CONFIG_EXECUTABLE}" --prefix
          OUTPUT_VARIABLE GSL_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        message( STATUS "Using GSL from ${GSL_PREFIX}")
      endif()
    else( GSL_CONFIG_EXECUTABLE )
      message( STATUS "FindGSL: gsl-config not found.")
    endif( GSL_CONFIG_EXECUTABLE )
  endif( UNIX OR MSYS )
endif( WIN32 AND NOT CYGWIN AND NOT MSYS )
 
if( GSL_FOUND )
  if( NOT GSL_FIND_QUIETLY )
    message( STATUS "Found GSL: ${GSL_INCLUDE_DIRS} ${GSL_LIBRARIES}" )
  endif( NOT GSL_FIND_QUIETLY )
else( GSL_FOUND )
  if( GSL_FIND_REQUIRED )
    message( FATAL_ERROR "FindGSL: Could not find GSL headers or library" )
  endif( GSL_FIND_REQUIRED )
endif( GSL_FOUND )

mark_as_advanced(
  GSL_CONFIG_EXECUTABLE
  GSL_INCLUDE_DIR
  GSL_LIBRARY
  GSL_CBLAS_LIBRARY
)


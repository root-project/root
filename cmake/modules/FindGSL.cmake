# Try to find gnu scientific library GSL
# See
# http://www.gnu.org/software/gsl/  and
# http://gnuwin32.sourceforge.net/packages/gsl.htm
#
# Based on a script of Felix Woelk and Jan Woetzel
# (www.mip.informatik.uni-kiel.de)
#
# It defines the following variables:
#  GSL_FOUND         - system has GSL lib
#  GSL_INCLUDE_DIR   - where to find headers
#  GSL_LIBRARY       - the gsl library
#  GSL_CBLAS_LIBRARY - the gslcblas library
#  GSL_LIBRARIES     - full path to the libraries
#  GSL_LIBRARY_DIR   - the directory where the PLplot library is found.
#  GSL_CFLAGS        - additional c (c++) required


# Windows, but not for Cygwin and MSys where gsl-config is available
if(WIN32 AND NOT CYGWIN AND NOT MSYS)
  # look for headers
  find_path(GSL_INCLUDE_DIR
    NAMES gsl/gsl_cdf.h gsl/gsl_randist.h
    PATHS ${GSL_DIR}/include $ENV{GSL_DIR}/include
  )

  if(GSL_INCLUDE_DIR)
    file(READ ${GSL_INCLUDE_DIR}/gsl/gsl_version.h versionstr)
    string(REGEX REPLACE ".*GSL_VERSION[ \"]*([0-9]+[.][0-9]+).*" "\\1" GSL_VERSION "${versionstr}")
  endif()

  # look for gsl library
  find_library(GSL_LIBRARY
    NAMES gsl
    PATHS ${GSL_DIR}/lib $ENV{GSL_DIR}/lib
  )

  if(GSL_LIBRARY)
    get_filename_component(GSL_LIBRARY_DIR ${GSL_LIBRARY} PATH )
  endif()

  # look for gsl cblas library
  find_library(GSL_CBLAS_LIBRARY
    NAMES gslcblas
    PATHS ${GSL_DIR}/lib $ENV{GSL_DIR}/lib
  )

  if(GSL_CBLAS_LIBRARY)
    set(GSL_LIBRARIES ${GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})
  else()
    set(GSL_LIBRARIES ${GSL_LIBRARY})
  endif()

  execute_process ( COMMAND lib /list ${GSL_LIBRARY} OUTPUT_VARIABLE content )
  string(FIND ${content} ".dll" APOSITION )
  if( NOT ("${APOSITION}" STREQUAL "-1") )
    set( GSL_CFLAGS "-DGSL_DLL" )
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(GSL REQUIRED_VARS GSL_INCLUDE_DIR GSL_LIBRARY VERSION_VAR GSL_VERSION)
  mark_as_advanced(GSL_INCLUDE_DIR GSL_LIBRARY GSL_CBLAS_LIBRARY)

else( WIN32 AND NOT CYGWIN AND NOT MSYS )
  if( UNIX OR MSYS )
    find_program( GSL_CONFIG_EXECUTABLE gsl-config
      ${GSL_DIR}/bin
      $ENV{GSL_DIR}/bin
      /usr/bin/
      /usr/local/bin
    )

    if( GSL_CONFIG_EXECUTABLE )

      # run the gsl-config program to get cxxflags
      execute_process(
        COMMAND sh "${GSL_CONFIG_EXECUTABLE}" --cflags
        OUTPUT_VARIABLE GSL_CFLAGS
        RESULT_VARIABLE RET
        ERROR_QUIET
      )
      if(RET EQUAL 0)
        string(STRIP "${GSL_CFLAGS}" GSL_CFLAGS)
        separate_arguments(GSL_CFLAGS)

        # parse definitions from cflags; drop -D* from CFLAGS
        string(REGEX MATCHALL "-D[^;]+" GSL_DEFINITIONS  "${GSL_CFLAGS}")
        string(REGEX REPLACE "-D[^;]+;" "" GSL_CFLAGS "${GSL_CFLAGS}")

        # parse include dirs from cflags; drop -I prefix
        string(REGEX MATCH "-I[^;]+" GSL_INCLUDE_DIR "${GSL_CFLAGS}")
        string(REPLACE "-I" "" GSL_INCLUDE_DIR "${GSL_INCLUDE_DIR}")
        string(REGEX REPLACE "-I[^;]+;" "" GSL_CFLAGS "${GSL_CFLAGS}")
      endif()

      # run the gsl-config program to get the libs
      execute_process(
        COMMAND sh "${GSL_CONFIG_EXECUTABLE}" --libs
        OUTPUT_VARIABLE GSL_LIBRARIES
        RESULT_VARIABLE RET
        ERROR_QUIET
      )
      if(RET EQUAL 0)
        string(STRIP "${GSL_LIBRARIES}" GSL_LIBRARIES)
        separate_arguments(GSL_LIBRARIES)

        # extract linkdirs (-L) for rpath (i.e., LINK_DIRECTORIES)
        string(REGEX MATCH "-L[^;]+" GSL_LIBRARY_DIR "${GSL_LIBRARIES}")
        string(REPLACE "-L" "" GSL_LIBRARY_DIR "${GSL_LIBRARY_DIR}")
      endif()

      execute_process(
        COMMAND sh "${GSL_CONFIG_EXECUTABLE}" --version
        OUTPUT_VARIABLE GSL_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )

      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(GSL REQUIRED_VARS GSL_INCLUDE_DIR GSL_LIBRARIES VERSION_VAR GSL_VERSION)

    else( GSL_CONFIG_EXECUTABLE )
      message( STATUS "FindGSL: gsl-config not found.")
    endif( GSL_CONFIG_EXECUTABLE )

    mark_as_advanced(GSL_CONFIG_EXECUTABLE)
    
  endif( UNIX OR MSYS )
endif( WIN32 AND NOT CYGWIN AND NOT MSYS )

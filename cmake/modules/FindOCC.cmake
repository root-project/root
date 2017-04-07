# - Try to find OpenCASCADE libraries
### Does not test what version has been found,though
### that could be done by parsing Standard_Version.hxx

# Once done, this will define
#  OCC_FOUND - true if OCC has been found
#  OCC_INCLUDE_DIR - the OCC include dir
#  OCC_LIBRARIES (not cached) - full path of OCC libraries

set(_occdirs ${CASROOT} ${CASS_DIR} $ENV{CASROOT} /opt/occ)

find_path(OCC_INCLUDE_DIR
          NAMES Standard_Real.hxx
          HINTS ${_occdirs} /usr/include/opencascade /usr/include/oce
          PATH_SUFFIXES inc
          DOC "Specify the directory containing Standard_Real.hxx")

foreach(_libname ${OCC_FIND_COMPONENTS})
  list(APPEND OCC_REQUIRED_LIBRARIES OCC_${_libname}_LIBRARY)
  find_library(OCC_${_libname}_LIBRARY $
              NAMES ${_libname}
              HINTS ${_occdirs}
              PATH_SUFFIXES lib)
  if(OCC_${_libname}_LIBRARY)
    list(APPEND OCC_LIBRARIES ${OCC_${_libname}_LIBRARY})
  endif()
endforeach()

# handle the QUIETLY and REQUIRED arguments and set OCC_FOUND to TRUE if
# all listed variables are TRUE

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OCC DEFAULT_MSG OCC_INCLUDE_DIR ${OCC_REQUIRED_LIBRARIES})
mark_as_advanced(OCC_INCLUDE_DIR ${OCC_REQUIRED_LIBRARIES})

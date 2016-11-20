#  Try to find CASTOR 
#  Check for rfio_api.h for CASTOR 2 and libshift
#  Defines:
#  CASTOR_INCLUDE_DIR - where to find rfio_api.h, etc.
#  CASTOR_<component>_LIBRARY 
#  CASTOR_<component>_FOUND  
#  CASTOR_LIBRARIES   - List of libraries when using castor (not cached)
#  CASTOR_VERSION     - Castor version (not cached)
#  CASTOR_FOUND       - True if CASTOR libraries found.

# Enforce a minimal list of libraries
list(APPEND CASTOR_FIND_COMPONENTS shift rfio common client ns)
list(REMOVE_DUPLICATES CASTOR_FIND_COMPONENTS)

set(locations
    ${CASTOR} $ENV{CASTOR}
    ${CASTOR_DIR} $ENV{CASTOR_DIR}
    /cern/pro
    /cern/new
    /cern/old
    /opt/shift)

find_path(CASTOR_INCLUDE_DIR rfio_api.h HINTS ${locations} PATH_SUFFIXES include include/shift)

# Obtain the version
if(CASTOR_INCLUDE_DIR)
  file(READ ${CASTOR_INCLUDE_DIR}/patchlevel.h contents)
  string(REGEX MATCH   "BASEVERSION[ ]*[\"][ ]*([^ \"]+)" cont ${contents})
  string(REGEX REPLACE "BASEVERSION[ ]*[\"][ ]*([^ \"]+)" "\\1" CASTOR_VERSION ${cont})
endif()

# Find all the libraires
foreach(component ${CASTOR_FIND_COMPONENTS})
  if(component STREQUAL shift) # libshift.so is the only one without the prefix 'castor'
    set(name ${component})
  else()
    set(name castor${component})
  endif()
  find_library(CASTOR_${component}_LIBRARY NAMES ${name} HINTS ${locations} PATH_SUFFIXES lib lib/shift)
  if (CASTOR_${component}_LIBRARY)
    set(CASTOR_${component}_FOUND 1)
    list(APPEND CASTOR_LIBRARIES ${CASTOR_${component}_LIBRARY})
  endif()
  mark_as_advanced(CASTOR_${component}_LIBRARY)
endforeach()

# Handle the QUIETLY and REQUIRED arguments and set CASTOR_FOUND to TRUE if all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CASTOR DEFAULT_MSG CASTOR_shift_LIBRARY CASTOR_INCLUDE_DIR)
mark_as_advanced(CASTOR_INCLUDE_DIR)

# - Locate pythia6 library
# Defines:
#
#  PYTHIA6_FOUND
#  PYTHIA6_INCLUDE_DIR
#  PYTHIA6_INCLUDE_DIRS (not cached)
#  PYTHIA6_LIBRARY
#  PYTHIA6_LIBRARY_DIR (not cached)
#  PYTHIA6_LIBRARIES (not cached)

set(_pythia6dirs
  ${PYTHIA6}
  $ENV{PYTHIA6}
  ${PYTHIA6_DIR}
  $ENV{PYTHIA6_DIR}
  /cern/pro/lib
  /opt/pythia
  /opt/pythia6
  /usr/lib/pythia
  /usr/local/lib/pythia
  /usr/lib/pythia6
  /usr/local/lib/pythia6)

find_path(PYTHIA6_INCLUDE_DIR general_pythia.inc
           HINTS ${_pythia6dirs}
           PATH_SUFFIXES include
           DOC "Specify the Pythia6 include dir here.")

if(PYTHIA6_INCLUDE_DIR)
  set(PYTHIA6_INCLUDE_DIRS ${PYTHIA6_INCLUDE_DIR})
endif()

find_library(PYTHIA6_LIBRARY NAMES pythia6 Pythia6
             HINTS ${_pythia6dirs}
             PATH_SUFFIXES lib
             DOC "Specify the Pythia6 library here.")

set(PYTHIA6_LIBRARIES ${PYTHIA6_LIBRARY})

foreach(lib rootinterface pythia6_dummy)
  find_library(PYTHIA6_${lib}_LIBRARY NAMES ${lib}
             HINTS ${_pythia6dirs}
             PATH_SUFFIXES lib
             DOC "Specify the Pythia rootinterface library here.")
  if(PYTHIA6_${lib}_LIBRARY)
    list(APPEND PYTHIA6_LIBRARIES ${PYTHIA6_${lib}_LIBRARY})
  endif()
endforeach()

get_filename_component(PYTHIA6_LIBRARY_DIR ${PYTHIA6_LIBRARY} PATH)

# handle the QUIETLY and REQUIRED arguments and set PHOTOS_FOUND to TRUE if
# all listed variables are TRUE

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pythia6 DEFAULT_MSG PYTHIA6_LIBRARY)

mark_as_advanced(PYTHIA6_INCLUDE_DIR
                 PYTHIA6_LIBRARY
                 PYTHIA6_rootinterface_LIBRARY)

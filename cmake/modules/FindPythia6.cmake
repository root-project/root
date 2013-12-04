# - Locate pythia6 library
# Defines:
#
#  PYTHIA6_FOUND
#  PYTHIA6_INCLUDE_DIR
#  PYTHIA6_INCLUDE_DIRS (not cached)
#  PYTHIA6_LIBRARY
#  PYTHIA6_LIBRARY_DIR (not cached)
#  PYTHIA6_LIBRARIES (not cached)

set(CMAKE_LIBRARY_PATH
  /cern/pro/lib
  /opt/pythia 
  /opt/pythia6
  /usr/lib/pythia
  /usr/local/lib/pythia
  /usr/lib/pythia6
  /usr/local/lib/pythia6
  /usr/lib
  /usr/local/lib)


find_library(PYTHIA6_LIBRARY NAMES pythia6 Pythia6
             HINTS $ENV{PYTHIA6_DIR}/lib ${PYTHIA6_DIR}/lib)

set(PYTHIA6_LIBRARIES ${PYTHIA6_LIBRARY})
get_filename_component(PYTHIA6_LIBRARY_DIR ${PYTHIA6_LIBRARY} PATH)

# handle the QUIETLY and REQUIRED arguments and set PHOTOS_FOUND to TRUE if
# all listed variables are TRUE

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Pythia6 DEFAULT_MSG PYTHIA6_LIBRARY)

mark_as_advanced(PYTHIA6_FOUND PYTHIA6_LIBRARY)

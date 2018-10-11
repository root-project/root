# Find the Pythia8 includes and library.
#
# This module defines
# PYTHIA8_INCLUDE_DIR   where to locate Pythia.h file
# PYTHIA8_LIBRARY       where to find the libpythia8 library
# PYTHIA8_<lib>_LIBRARY Addicional libraries
# PYTHIA8_LIBRARIES     (not cached) the libraries to link against to use Pythia8
# PYTHIA8_FOUND         if false, you cannot build anything that requires Pythia8
# PYTHIA8_VERSION       version of Pythia8 if found

set(_pythia8dirs
    ${PYTHIA8}
    $ENV{PYTHIA8}
    ${PYTHIA8_DIR}
    $ENV{PYTHIA8_DIR}
    /usr
    /opt/pythia8)

find_path(PYTHIA8_INCLUDE_DIR
          NAMES Pythia8/Pythia.h
          HINTS ${_pythia8dirs}
          PATH_SUFFIXES include include/Pythia8 include/pythia8
          DOC "Specify the directory containing Pythia.h.")

find_library(PYTHIA8_LIBRARY
             NAMES pythia8 Pythia8
             HINTS ${_pythia8dirs}
             PATH_SUFFIXES lib
             DOC "Specify the Pythia8 library here.")

find_library(PYTHIA8_hepmcinterface_LIBRARY
             NAMES hepmcinterface pythia8tohepmc
             HINTS ${_pythia8dirs}
             PATH_SUFFIXES lib)

find_library(PYTHIA8_lhapdfdummy_LIBRARY
             NAMES lhapdfdummy
             HINTS ${_pythia8dirs}
             PATH_SUFFIXES lib)

foreach(_lib PYTHIA8_LIBRARY PYTHIA8_hepmcinterface_LIBRARY PYTHIA8_lhapdfdummy_LIBRARY)
  if(${_lib})
    set(PYTHIA8_LIBRARIES ${PYTHIA8_LIBRARIES} ${${_lib}})
  endif()
endforeach()
set(PYTHIA8_INCLUDE_DIRS ${PYTHIA8_INCLUDE_DIR} ${PYTHIA8_INCLUDE_DIR}/Pythia8 )

find_path(PYTHIA8_DATA 
          NAMES MainProgramSettings.xml
          HINTS ${_pythia8dirs}
          PATH_SUFFIXES xmldoc)

# handle the QUIETLY and REQUIRED arguments and set PYTHIA8_FOUND to TRUE if
# all listed variables are TRUE

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Pythia8 DEFAULT_MSG PYTHIA8_INCLUDE_DIR PYTHIA8_LIBRARY)
mark_as_advanced(PYTHIA8_INCLUDE_DIR PYTHIA8_LIBRARY PYTHIA8_hepmcinterface_LIBRARY PYTHIA8_lhapdfdummy_LIBRARY)

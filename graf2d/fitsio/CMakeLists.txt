############################################################################
# CMakeLists.txt file for building ROOT graf2d/fitsio package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(FITSIO
  HEADERS
    TFITS.h
  SOURCES
    TFITS.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  LIBRARIES
    ${CFITSIO_LIBRARIES}
  DEPENDENCIES
    Gpad
    Graf
    Hist
    Matrix
  BUILTINS
    CFITSIO
)

target_include_directories(FITSIO PRIVATE ${CFITSIO_INCLUDE_DIR})

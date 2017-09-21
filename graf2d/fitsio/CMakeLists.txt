############################################################################
# CMakeLists.txt file for building ROOT graf2d/fitsio package
# @author Pere Mato, CERN
############################################################################

include_directories(${CFITSIO_INCLUDE_DIR})

ROOT_STANDARD_LIBRARY_PACKAGE(FITSIO
                              HEADERS *.h
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              LIBRARIES ${CFITSIO_LIBRARIES}
                              DEPENDENCIES Hist Gpad Graf Matrix BUILTINS CFITSIO)


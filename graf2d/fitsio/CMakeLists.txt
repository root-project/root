############################################################################
# CMakeLists.txt file for building ROOT graf2d/fitsio package
# @author Pere Mato, CERN
############################################################################

include_directories(${CFITSIO_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__FITSIO *.h MODULE FITSIO LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Hist Gpad Graf BUILTINS CFITSIO)

ROOT_LINKER_LIBRARY(FITSIO *.cxx G__FITSIO.cxx LIBRARIES ${CFITSIO_LIBRARIES} DEPENDENCIES Hist Gpad Graf Matrix BUILTINS CFITSIO)

ROOT_INSTALL_HEADERS()

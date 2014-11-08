############################################################################
# CMakeLists.txt file for building ROOT graf2d/fitsio package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(graf2d/graf)
ROOT_USE_PACKAGE(hist/hist)

include_directories(${CFITSIO_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__FITSIO *.h MODULE FITSIO LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(FITSIO *.cxx G__FITSIO.cxx LIBRARIES ${CFITSIO_LIBRARIES} DEPENDENCIES Hist Gpad Graf Matrix)
if(builtin_cfitsio)
   ROOT_ADD_BUILTIN_DEPENDENCIES(FITSIO CFITSIO)
endif()
ROOT_INSTALL_HEADERS()



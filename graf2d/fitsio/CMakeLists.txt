############################################################################
# CMakeLists.txt file for building ROOT graf2d/fitsio package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(graf2d/graf)
ROOT_USE_PACKAGE(hist/hist)

include_directories(${CFITSIO_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__FITSIO *.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(FITSIO LINKDEF LinkDef.h DEPENDENCIES Graf)
ROOT_LINKER_LIBRARY(FITSIO *.cxx G__FITSIO.cxx LIBRARIES ${CFITSIO_LIBRARIES} DEPENDENCIES Graf)
ROOT_INSTALL_HEADERS()



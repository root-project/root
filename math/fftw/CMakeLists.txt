############################################################################
# CMakeLists.txt file for building ROOT math/fftw package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(math/mathcore)
include_directories(${FFTW_INCLUDE_DIR})

ROOT_GENERATE_DICTIONARY(G__FFTW *.h MODULE FFTW LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(FFTW *.cxx G__FFTW.cxx LIBRARIES Core ${FFTW_LIBRARIES} DEPENDENCIES )
ROOT_INSTALL_HEADERS()

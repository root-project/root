############################################################################
# CMakeLists.txt file for building ROOT montecarlo/pythia8 package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(montecarlo/eg)
ROOT_USE_PACKAGE(math/physics)
include_directories(${PYTHIA8_INCLUDE_DIR})

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-overloaded-virtual)

ROOT_GENERATE_DICTIONARY(G__Pythia8 *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(EGPythia8 LINKDEF LinkDef.h DEPENDENCIES EG Graf VMC Physics )
ROOT_LINKER_LIBRARY(EGPythia8 *.cxx G__Pythia8.cxx LIBRARIES Core ${PYTHIA8_LIBRARIES} DEPENDENCIES EG Graf VMC Physics)
ROOT_INSTALL_HEADERS()

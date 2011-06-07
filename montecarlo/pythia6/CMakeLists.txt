############################################################################
# CMakeLists.txt file for building ROOT montecarlo/pythia6 package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(montecarlo/eg)
ROOT_USE_PACKAGE(montecarlo/vmc)
ROOT_USE_PACKAGE(math/physics)

ROOT_GENERATE_DICTIONARY(G__Pythia6 *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(EGPythia6 LINKDEF LinkDef.h DEPENDENCIES EG Graf VMC Physics )
ROOT_LINKER_LIBRARY(EGPythia6 *.cxx G__Pythia6.cxx LIBRARIES Core ${PYTHIA6_LIBRARIES} DEPENDENCIES EG Graf VMC Physics)
ROOT_INSTALL_HEADERS()

############################################################################
# CMakeLists.txt file for building ROOT math/genetic package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(tmva)

ROOT_GENERATE_DICTIONARY(G__Genetic Math/GeneticMinimizer.h MODULE Genetic LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(Genetic *.cxx G__Genetic.cxx LIBRARIES Core DEPENDENCIES MathCore TMVA)

ROOT_INSTALL_HEADERS()

ROOT_ADD_TEST_SUBDIRECTORY(test)

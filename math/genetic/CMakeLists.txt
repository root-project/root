############################################################################
# CMakeLists.txt file for building ROOT math/genetic package
# @author Pere Mato, CERN
############################################################################

ROOT_USE_PACKAGE(math/mathcore)
ROOT_USE_PACKAGE(tmva)

ROOT_GENERATE_DICTIONARY(G__Genetic Math/GeneticMinimizer.h LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(Genetic LINKDEF LinkDef.h DEPENDENCIES RIO Hist Matrix Tree Graf Gpad TreePlayer MLP Minuit MathCore XMLIO TMVA)
ROOT_LINKER_LIBRARY(Genetic *.cxx G__Genetic.cxx LIBRARIES Core Cint DEPENDENCIES MathCore TMVA)

ROOT_INSTALL_HEADERS()

############################################################################
# CMakeLists.txt file for building ROOT R package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RMVA
  HEADERS
    TMVA/MethodC50.h
    TMVA/MethodRSNNS.h
    TMVA/MethodRSVM.h
    TMVA/MethodRXGB.h
    TMVA/RMethodBase.h
  SOURCES
    src/MethodC50.cxx
    src/MethodRSNNS.cxx
    src/MethodRSVM.cxx
    src/MethodRXGB.cxx
    src/RMethodBase.cxx
  DEPENDENCIES
    Core
    Matrix
    RInterface
    RIO
    Thread
    TMVA
)

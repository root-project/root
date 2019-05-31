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
  DICTIONARY_OPTIONS
    -I${R_INCLUDE_DIR}
  DEPENDENCIES
    Core
    Matrix
    RInterface
    RIO
    Thread
    TMVA
)

target_include_directories(RMVA BEFORE PRIVATE ${R_INCLUDE_DIR})

############################################################################
# CMakeLists.txt file for building ROOT math/foam package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Foam
  HEADERS
    TFoam.h
    TFoamCell.h
    TFoamIntegrand.h
    TFoamMaxwt.h
    TFoamSampler.h
    TFoamVect.h
  SOURCES
    src/TFoam.cxx
    src/TFoamCell.cxx
    src/TFoamIntegrand.cxx
    src/TFoamMaxwt.cxx
    src/TFoamSampler.cxx
    src/TFoamVect.cxx
  DEPENDENCIES
    Hist
    MathCore
)

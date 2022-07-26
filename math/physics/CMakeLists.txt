# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/physics package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Physics
  HEADERS
    TFeldmanCousins.h
    TGenPhaseSpace.h
    TLorentzRotation.h
    TLorentzVector.h
    TQuaternion.h
    TRobustEstimator.h
    TRolke.h
    TRotation.h
    TVector2.h
    TVector3.h
  SOURCES
    src/TFeldmanCousins.cxx
    src/TGenPhaseSpace.cxx
    src/TLorentzRotation.cxx
    src/TLorentzVector.cxx
    src/TQuaternion.cxx
    src/TRobustEstimator.cxx
    src/TRolke.cxx
    src/TRotation.cxx
    src/TVector2.cxx
    src/TVector3.cxx
  DEPENDENCIES
    Matrix
    MathCore
    GenVector
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
)

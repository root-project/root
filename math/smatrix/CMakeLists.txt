# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/smatrix package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Smatrix
  HEADERS
    Math/BinaryOperators.h
    Math/BinaryOpPolicy.h
    Math/CholeskyDecomp.h
    Math/Dfact.h
    Math/Dfactir.h
    Math/Dfinv.h
    Math/Dinv.h
    Math/Dsfact.h
    Math/Dsinv.h
    Math/Expression.h
    Math/Functions.h
    Math/HelperOps.h
    Math/MatrixFunctions.h
    Math/MatrixRepresentationsStatic.h
    Math/MConfig.h
    Math/SMatrixDfwd.h
    Math/SMatrixFfwd.h
    Math/SMatrix.h
    Math/StaticCheck.h
    Math/SVector.h
    Math/UnaryOperators.h
  NO_SOURCES
  LINKDEF
    LinkDefAll.h
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    MathCore
)

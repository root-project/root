# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/smatrix package
############################################################################

set(SMATRIX_HEADERS
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
)

ROOT_STANDARD_LIBRARY_PACKAGE(Smatrix
  HEADERS
    ${SMATRIX_HEADERS}
  NO_SOURCES
  LINKDEF
    LinkDef.h
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    MathCore
)

set(RELATIVE_INC_HEADERS ${SMATRIX_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Smatrix
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    inc/Math/CramerInversion.icc
    inc/Math/CramerInversionSym.icc
    inc/Math/MatrixInversion.icc
    inc/Math/SMatrix.icc
    inc/Math/SVector.icc
)

set(SMATRIX32_HEADERS
  Math/SMatrix.h
  Math/SMatrixDfwd.h
  Math/SMatrixFfwd.h
  Math/SVector.h
)

ROOT_GENERATE_DICTIONARY(G__Smatrix32
  ${SMATRIX32_HEADERS}
  MULTIDICT
  MODULE
    Smatrix
  LINKDEF
    LinkDefD32.h
  OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    MathCore
)

set(RELATIVE_INC_HEADERS ${SMATRIX32_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(G__Smatrix32
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDefD32.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)

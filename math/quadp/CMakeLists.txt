# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/quadp package
############################################################################

set(QUADP_HEADERS
  TGondzioSolver.h
  TMehrotraSolver.h
  TQpDataBase.h
  TQpDataDens.h
  TQpDataSparse.h
  TQpLinSolverBase.h
  TQpLinSolverDens.h
  TQpLinSolverSparse.h
  TQpProbBase.h
  TQpProbDens.h
  TQpProbSparse.h
  TQpResidual.h
  TQpSolverBase.h
  TQpVar.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Quadp
  HEADERS
    ${QUADP_HEADERS}
  SOURCES
    src/TGondzioSolver.cxx
    src/TMehrotraSolver.cxx
    src/TQpDataBase.cxx
    src/TQpDataDens.cxx
    src/TQpDataSparse.cxx
    src/TQpLinSolverBase.cxx
    src/TQpLinSolverDens.cxx
    src/TQpLinSolverSparse.cxx
    src/TQpProbBase.cxx
    src/TQpProbDens.cxx
    src/TQpProbSparse.cxx
    src/TQpResidual.cxx
    src/TQpSolverBase.cxx
    src/TQpVar.cxx
  DEPENDENCIES
    Matrix
)

set(RELATIVE_INC_HEADERS ${QUADP_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Quadp
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

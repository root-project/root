# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT math/quadp package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Quadp
  HEADERS
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

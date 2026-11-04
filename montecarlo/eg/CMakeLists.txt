# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT montecarlo/eg package
# @author Pere Mato, CERN
############################################################################

set(EG_HEADERS
  TAttParticle.h
  TDatabasePDG.h
  TDecayChannel.h
  TGenerator.h
  TPDGCode.h
  TParticleClassPDG.h
  TParticle.h
  TParticlePDG.h
  TPrimary.h
  TVirtualMCDecayer.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(EG
  HEADERS
    ${EG_HEADERS}
  SOURCES
    src/TAttParticle.cxx
    src/TDatabasePDG.cxx
    src/TDecayChannel.cxx
    src/TGenerator.cxx
    src/TParticleClassPDG.cxx
    src/TParticle.cxx
    src/TParticlePDG.cxx
    src/TPrimary.cxx
    src/TVirtualMCDecayer.cxx
  DEPENDENCIES
    Gpad
    Graf
    Graf3d
    MathCore
    Physics
  LIBRARIES
    ROOTX3dViewer
)

set(RELATIVE_INC_HEADERS ${EG_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(EG
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    inc/Hepevt.h
    inc/cfortran.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)

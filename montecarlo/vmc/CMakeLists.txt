# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT montecarlo/vmc package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(VMC
  HEADERS
    TGeoMCGeometry.h
    TMCAutoLock.h
    TMCOptical.h
    TMCParticleType.h
    TMCProcess.h
    TMCtls.h
    TMCVerbose.h
    TVirtualMCApplication.h
    TVirtualMCGeometry.h
    TVirtualMC.h
    TVirtualMCSensitiveDetector.h
    TVirtualMCStack.h
    TMCManager.h
    TMCManagerStack.h
    TGeoMCBranchArrayContainer.h
    TMCParticleStatus.h
  SOURCES
    src/TGeoMCGeometry.cxx
    src/TMCAutoLock.cxx
    src/TMCVerbose.cxx
    src/TVirtualMCApplication.cxx
    src/TVirtualMC.cxx
    src/TVirtualMCGeometry.cxx
    src/TVirtualMCSensitiveDetector.cxx
    src/TVirtualMCStack.cxx
    src/TMCManager.cxx
    src/TMCManagerStack.cxx
    src/TGeoMCBranchArrayContainer.cxx
  DEPENDENCIES
    EG
    Geom
    MathCore
    Physics
)

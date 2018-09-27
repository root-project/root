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
    TPDGCode.h
    TVirtualMCApplication.h
    TVirtualMCGeometry.h
    TVirtualMC.h
    TVirtualMCSensitiveDetector.h
    TVirtualMCStack.h
  SOURCES
    src/TGeoMCGeometry.cxx
    src/TMCAutoLock.cxx
    src/TMCVerbose.cxx
    src/TVirtualMCApplication.cxx
    src/TVirtualMC.cxx
    src/TVirtualMCGeometry.cxx
    src/TVirtualMCSensitiveDetector.cxx
    src/TVirtualMCStack.cxx
  DEPENDENCIES
    EG
    Geom
    MathCore
)

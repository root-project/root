############################################################################
# CMakeLists.txt file for building ROOT montecarlo/eg package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(EG
  HEADERS
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
)

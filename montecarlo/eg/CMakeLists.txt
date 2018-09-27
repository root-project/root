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
    TParticleClassPDG.h
    TParticle.h
    TParticlePDG.h
    TPrimary.h
    TVirtualMCDecayer.h
  SOURCES
    TAttParticle.cxx
    TDatabasePDG.cxx
    TDecayChannel.cxx
    TGenerator.cxx
    TParticleClassPDG.cxx
    TParticle.cxx
    TParticlePDG.cxx
    TPrimary.cxx
    TVirtualMCDecayer.cxx
  DEPENDENCIES
    Gpad
    Graf
    Graf3d
    MathCore
)

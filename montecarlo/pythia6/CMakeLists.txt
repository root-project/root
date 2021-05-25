# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT montecarlo/pythia6 package
# @author Pere Mato, CERN
############################################################################

if(pythia6_nolink)
  set(PYTHIA6_LIBRARIES)
  if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    string(REGEX REPLACE "-Wl,--no-undefined" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")
  endif()
else()
  if(MSVC)
    SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /SAFESEH:NO ")
    link_directories($ENV{LIB})
  endif()
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(EGPythia6
  HEADERS
    TMCParticle.h
    TPythia6Calls.h
    TPythia6Decayer.h
    TPythia6.h
  SOURCES
    src/TMCParticle.cxx
    src/TPythia6.cxx
    src/TPythia6Decayer.cxx
  LIBRARIES
    ${PYTHIA6_LIBRARIES}
  DEPENDENCIES
    Core
    EG
    Graf
    Physics
)

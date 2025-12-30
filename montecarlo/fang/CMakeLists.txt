# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.

############################################################################
# CMakeLists.txt file for building ROOT montecarlo/fang package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(FANG
  HEADERS
    FANG.h
  SOURCES
    src/FANG.cxx
  DEPENDENCIES
    Core
    MathCore
    Physics
    GenVector
)

if(testing)
  add_subdirectory(test)
endif()
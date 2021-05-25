# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT misc/memstat package
# @author Pere Mato, CERN
############################################################################

ROOT_ADD_CXX_FLAG(CMAKE_CXX_FLAGS -Wno-deprecated-declarations)

set(sources TMemStat.cxx TMemStatMng.cxx TMemStatBacktrace.cxx TMemStatHelpers.cxx TMemStatHook.cxx)
set(headers TMemStatHelpers.h TMemStat.h TMemStatBacktrace.h TMemStatDef.h TMemStatMng.h TMemStatHook.h )

ROOT_STANDARD_LIBRARY_PACKAGE(MemStat
                              HEADERS ${headers}
                              SOURCES ${sources}
                              LIBRARIES ${CMAKE_DL_LIBS}
                              DEPENDENCIES Tree Gpad Graf)

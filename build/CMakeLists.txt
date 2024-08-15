# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT build package
# @author Pere Mato, CERN
############################################################################

if(WIN32)
  ROOT_EXECUTABLE(bindexplib
    win/bindexplib/bindexplib.cxx
  )
  file(COPY win/w32pragma.h DESTINATION ${CMAKE_BINARY_DIR}/include/)
  file(COPY win/sehmap.h DESTINATION ${CMAKE_BINARY_DIR}/include/)
endif()

ROOT_EXECUTABLE(rmkdepend
  rmkdepend/cppsetup.c
  rmkdepend/ifparser.c
  rmkdepend/include.c
  rmkdepend/main.c
  rmkdepend/mainroot.cxx
  rmkdepend/parse.c
  rmkdepend/pr.c
)

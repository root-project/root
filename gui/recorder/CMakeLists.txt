# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/recorder package
############################################################################

set(RECORDER_HEADERS
  TRecorder.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Recorder
  HEADERS
    ${RECORDER_HEADERS}
  SOURCES
    src/TRecorder.cxx
  DEPENDENCIES
    RIO
    Tree
    Gui
    Thread
)

set(RELATIVE_INC_HEADERS ${RECORDER_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(Recorder
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

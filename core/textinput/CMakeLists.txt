# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/textinout package
############################################################################

set(TextInput_dict_headers
  Getline.h
  PARENT_SCOPE
)

ROOT_OBJECT_LIBRARY(TextInput
  src/Getline_color.cxx
  src/Getline.cxx
  src/textinput/Editor.cpp
  src/textinput/History.cpp
  src/textinput/KeyBinding.cpp
  src/textinput/Range.cpp
  src/textinput/SignalHandler.cpp
  src/textinput/StreamReader.cpp
  src/textinput/StreamReaderUnix.cpp
  src/textinput/StreamReaderWin.cpp
  src/textinput/TerminalConfigUnix.cpp
  src/textinput/TerminalDisplay.cpp
  src/textinput/TerminalDisplayUnix.cpp
  src/textinput/TerminalDisplayWin.cpp
  src/textinput/TextInputContext.cpp
  src/textinput/TextInput.cpp
)

target_include_directories(TextInput PRIVATE
   ${PROJECT_SOURCE_DIR}/core/textinput/src
   ${PROJECT_SOURCE_DIR}/core/base/inc
   ${PROJECT_SOURCE_DIR}/core/clib/inc
   ${PROJECT_SOURCE_DIR}/core/meta/inc
   ${PROJECT_SOURCE_DIR}/core/cont/inc
   ${PROJECT_SOURCE_DIR}/core/gui/inc
   ${PROJECT_SOURCE_DIR}/core/thread/inc
   ${PROJECT_SOURCE_DIR}/core/foundation/inc
   ${PROJECT_BINARY_DIR}/ginclude
)

ROOT_INSTALL_HEADERS()

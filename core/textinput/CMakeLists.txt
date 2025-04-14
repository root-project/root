# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/textinput package
############################################################################

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS Getline.h)

target_sources(Core PRIVATE
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

target_include_directories(Core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  PRIVATE
    src
)

ROOT_INSTALL_HEADERS()

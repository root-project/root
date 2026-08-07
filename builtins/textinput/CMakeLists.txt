# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building textinput package for ROOT
############################################################################

target_sources(Core PRIVATE
  Editor.cpp
  History.cpp
  KeyBinding.cpp
  Range.cpp
  SignalHandler.cpp
  StreamReader.cpp
  StreamReaderUnix.cpp
  StreamReaderWin.cpp
  TerminalConfigUnix.cpp
  TerminalDisplay.cpp
  TerminalDisplayUnix.cpp
  TerminalDisplayWin.cpp
  TextInputContext.cpp
  TextInput.cpp
  UTF8.cpp
)

target_include_directories(Core
  PRIVATE
    ../ # to be able to call #include "textinput/header.h" from ROOT core files
)

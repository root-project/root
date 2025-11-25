# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building textinput package for ROOT
############################################################################

add_library(textinput STATIC)

target_sources(textinput PRIVATE
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
  PUBLIC
  FILE_SET HEADERS
  TYPE HEADERS
  BASE_DIRS ./ # to be able to call #include "textinput/header.h" from ROOT core files
  FILES
    textinput/Callbacks.h
    textinput/Callbacks.h
    textinput/Color.h
    textinput/Display.h
    textinput/Editor.h
    textinput/History.h
    textinput/InputData.h
    textinput/KeyBinding.h
    textinput/Range.h
    textinput/Reader.h
    textinput/SignalHandler.h
    textinput/StreamReader.h
    textinput/StreamReaderUnix.h
    textinput/StreamReaderWin.h
    textinput/TerminalConfigUnix.h
    textinput/TerminalDisplay.h
    textinput/TerminalDisplayUnix.h
    textinput/TerminalDisplayWin.h
    textinput/Text.h
    textinput/TextInputContext.h
    textinput/TextInput.h
)

set_property(TARGET textinput PROPERTY POSITION_INDEPENDENT_CODE ON)

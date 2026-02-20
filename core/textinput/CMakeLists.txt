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

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      Core
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/ src/
      FILES
          inc/Getline.h
          src/Getline_color.h
          src/textinput/Callbacks.h
          src/textinput/Callbacks.h
          src/textinput/Color.h
          src/textinput/Display.h
          src/textinput/Editor.h
          src/textinput/History.h
          src/textinput/InputData.h
          src/textinput/KeyBinding.h
          src/textinput/Range.h
          src/textinput/Reader.h
          src/textinput/SignalHandler.h
          src/textinput/StreamReader.h
          src/textinput/StreamReaderUnix.h
          src/textinput/StreamReaderWin.h
          src/textinput/TerminalConfigUnix.h
          src/textinput/TerminalDisplay.h
          src/textinput/TerminalDisplayUnix.h
          src/textinput/TerminalDisplayWin.h
          src/textinput/Text.h
          src/textinput/TextInputContext.h
          src/textinput/TextInput.h
  )
endif()

ROOT_INSTALL_HEADERS()

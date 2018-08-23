############################################################################
# CMakeLists.txt file for building ROOT core/textinout package
############################################################################

set(sources src/Getline_color.cxx
   src/Getline.cxx
   src/textinput/Editor.cpp
   src/textinput/Range.cpp
   src/textinput/StreamReaderUnix.cpp
   src/textinput/TerminalDisplay.cpp
   src/textinput/TextInputContext.cpp
   src/textinput/History.cpp
   src/textinput/SignalHandler.cpp
   src/textinput/StreamReaderWin.cpp
   src/textinput/TerminalDisplayUnix.cpp
   src/textinput/TextInput.cpp
   src/textinput/KeyBinding.cpp
   src/textinput/StreamReader.cpp
   src/textinput/TerminalConfigUnix.cpp
   src/textinput/TerminalDisplayWin.cpp
)
set(headers Getline.h)
set(TextInput_dict_headers ${headers} PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
ROOT_OBJECT_LIBRARY(TextInput ${sources})

ROOT_INSTALL_HEADERS()
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)

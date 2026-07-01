# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/qt6canvas package
############################################################################

find_package(Qt6 COMPONENTS Core Widgets CONFIG)

if(NOT Qt6_FOUND)
  if(fail-on-missing)
    message(FATAL_ERROR "Could NOT find Qt6 with Gui components")
  else()
    message(WARNING "Qt6 (Gui) not found, disabling option 'qt6canvas'")
    set(qt6canv OFF CACHE BOOL "Disabled because Qt6 Gui not found" FORCE)
    return()
  endif()
endif()


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTQt6Canvas
  HEADERS
    TQt6Canvas.h
    TQt6PadPainter.h
  SOURCES
    src/QCanvasWidget.cpp
    src/QPaintWidget.cpp
    src/QRootMethodDialog.cpp
    src/TQt6Canvas.cxx
    src/TQt6PadPainter.cxx
  LIBRARIES
    Qt6::Widgets
  DEPENDENCIES
    Core
    Gpad
)

target_include_directories(ROOTQt6Canvas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
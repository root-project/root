# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/qt6canvas package
############################################################################

if(NOT Qt6_FOUND)
  message(SEND_ERROR "Qt6 (Widgets) was not found, install missing components or disbale option 'qt6canv' ")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(QT6CANVAS_HEADERS
  TQt6Application.h
  TQt6Canvas.h
  TQt6PadPainter.h
  TQt6GedEditor.h
  TQt6GuiFactory.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTQt6Canvas
  HEADERS
    ${QT6CANVAS_HEADERS}
  SOURCES
    src/QCanvasWidget.cpp
    src/QPaintWidget.cpp
    src/QRootMethodDialog.cpp
    src/QRootContextMenu.cpp
    src/TQt6Application.cxx
    src/TQt6Canvas.cxx
    src/TQt6PadPainter.cxx
    src/TQt6GedEditor.cxx
    src/TQt6GuiFactory.cxx
  LIBRARIES
    Qt6::Core
    Qt6::Widgets
  DEPENDENCIES
    Core
    Gpad
)

set(RELATIVE_INC_HEADERS ${QT6CANVAS_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTQt6Canvas
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    src/QCanvasWidget.h
    src/QPaintWidget.h
    src/QRootContextMenu.h
    src/QRootMethodDialog.h
)

# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/gtk4canvas package
############################################################################

if(NOT GTKMM_FOUND)
  message(SEND_ERROR "Could NOT find gtkmm-4.0 components, install missing components or disable option 'gtk4canv'")
endif()

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTGtk4Canvas
  HEADERS
    TGtk4Canvas.h
    TGtk4GuiFactory.h
    TGtk4PadPainter.h
    TGtk4GedEditor.h
  SOURCES
    Gtk4DrawArea.cxx
    Gtk4CanvasWindow.cxx
    Gtk4MethodDialog.cxx
    TGtk4ContextMenu.cxx
    TGtk4Canvas.cxx
    TGtk4GuiFactory.cxx
    TGtk4PadPainter.cxx
    TGtk4GedEditor.cxx
  LIBRARIES
    PkgConfig::GTKMM
  DEPENDENCIES
    Core
    Gpad
)

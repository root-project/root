# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/meta package
############################################################################

set(GUI_HEADERS
  GuiTypes.h
  TApplicationImp.h
  TBrowser.h
  TBrowserImp.h
  TCanvasImp.h
  TClassMenuItem.h
  TContextMenu.h
  TContextMenuImp.h
  TControlBarImp.h
  TGuiFactory.h
  TInspectorImp.h
  TObjectSpy.h
  TToggleGroup.h
  TToggle.h
)



set_property(TARGET Core APPEND PROPERTY DICT_HEADERS ${GUI_HEADERS})

target_sources(Core PRIVATE
  src/InitGui.cxx
  src/TApplicationImp.cxx
  src/TBrowser.cxx
  src/TBrowserImp.cxx
  src/TCanvasImp.cxx
  src/TClassMenuItem.cxx
  src/TContextMenu.cxx
  src/TContextMenuImp.cxx
  src/TControlBarImp.cxx
  src/TGuiFactory.cxx
  src/TInspectorImp.cxx
  src/TObjectSpy.cxx
  src/TToggle.cxx
  src/TToggleGroup.cxx
)

target_include_directories(Core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

set(RELATIVE_INC_HEADERS ${GUI_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      Core
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/
      FILES
          ${RELATIVE_INC_HEADERS}
  )
endif()

ROOT_INSTALL_HEADERS()

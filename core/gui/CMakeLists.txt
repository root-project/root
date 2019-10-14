# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/meta package
############################################################################

set(GuiCore_dict_headers
  GuiTypes.h
  TBrowser.h
  TBrowserImp.h
  TCanvasImp.h
  TColor.h
  TColorGradient.h
  TContextMenu.h
  TContextMenuImp.h
  TControlBarImp.h
  TToggleGroup.h
  TToggle.h
  PARENT_SCOPE
)

ROOT_OBJECT_LIBRARY(GuiCore
  src/TBrowser.cxx
  src/TBrowserImp.cxx
  src/TCanvasImp.cxx
  src/TColor.cxx
  src/TColorGradient.cxx
  src/TContextMenu.cxx
  src/TContextMenuImp.cxx
  src/TControlBarImp.cxx
  src/TToggle.cxx
  src/TToggleGroup.cxx
)

# target_include_directories(GuiCore PRIVATE res ../foundation/res)

ROOT_INSTALL_HEADERS()

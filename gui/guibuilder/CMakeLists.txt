# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/guibuilder package
############################################################################

set(GUIBLD_HEADERS
  TGuiBldDragManager.h
  TGuiBldEditor.h
  TGuiBldGeometryFrame.h
  TGuiBldHintsButton.h
  TGuiBldHintsEditor.h
  TGuiBldNameFrame.h
  TRootGuiBuilder.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(GuiBld
  HEADERS
    ${GUIBLD_HEADERS}
  SOURCES
    src/TGuiBldDragManager.cxx
    src/TGuiBldEditor.cxx
    src/TGuiBldGeometryFrame.cxx
    src/TGuiBldHintsButton.cxx
    src/TGuiBldHintsEditor.cxx
    src/TGuiBldNameFrame.cxx
    src/TRootGuiBuilder.cxx
  DEPENDENCIES
    Gui
    Graf
    MathCore
    ASImageUtils
)

set(RELATIVE_INC_HEADERS ${GUIBLD_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(GuiBld
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

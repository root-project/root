############################################################################
# CMakeLists.txt file for building ROOT gui/guibuilder package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(GuiBld
  HEADERS
    TGuiBldDragManager.h
    TGuiBldEditor.h
    TGuiBldGeometryFrame.h
    TGuiBldHintsButton.h
    TGuiBldHintsEditor.h
    TGuiBldNameFrame.h
    TRootGuiBuilder.h
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
)

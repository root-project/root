############################################################################
# CMakeLists.txt file for building ROOT gui/webgui6 package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(WebGui6
  HEADERS
    TWebCanvas.h
    TWebGuiFactory.h
    TWebPadPainter.h
    TWebPainting.h
    TWebPS.h
    TWebSnapshot.h
  SOURCES
    src/TWebCanvas.cxx
    src/TWebGuiFactory.cxx
    src/TWebPadPainter.cxx
    src/TWebPainting.cxx
    src/TWebPS.cxx
    src/TWebSnapshot.cxx
  DEPENDENCIES
    Core
    Gui 
    ROOTWebDisplay
)

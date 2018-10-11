############################################################################
# CMakeLists.txt file for building ROOT gui/web package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(WebGui
  HEADERS
    TWebCanvas.h
    TWebGuiFactory.h
    TWebVirtualX.h
    TWebMenuItem.h
    TWebPadPainter.h
    TWebPainting.h
    TWebSnapshot.h
  SOURCES
    src/TWebCanvas.cxx
    src/TWebGuiFactory.cxx
    src/TWebVirtualX.cxx
    src/TWebMenuItem.cxx
    src/TWebPadPainter.cxx
    src/TWebPainting.cxx
    src/TWebSnapshot.cxx
  LIBRARIES
    Core
    Gui 
    ROOTWebDisplay
)



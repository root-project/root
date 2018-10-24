############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTWebDisplay
    HEADERS 
       ROOT/RWebBrowserArgs.hxx
       ROOT/RWebDisplayHandle.hxx
       ROOT/RWebWindow.hxx
       ROOT/RWebWindowsManager.hxx
    SOURCES
       RWebBrowserArgs.cxx
       RWebDisplayHandle.cxx
       RWebWindow.cxx
       RWebWindowsManager.cxx
    LIBRARIES 
       Core 
       RHTTP 
       MathCore
)
